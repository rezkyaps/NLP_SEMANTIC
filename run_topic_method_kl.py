import time
import logging
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import datetime
from sqlalchemy import create_engine
from collections import defaultdict
from scipy.spatial.distance import jensenshannon

# --- Logging setup ---
logging.basicConfig(
    filename="keyword_extraction.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Method semantic definitions ---
method_semantic_defs = {
    "01": "Positive reinforcement: This method introduces pleasant stimuli such as food treats, praise, toys, or playtime immediately after a desired behavior is performed. Its goal is to increase the frequency of that behavior by associating it with a rewarding consequence.",
    "02": "Negative punishment: This method removes a pleasant stimulus, such as attention, play, or treats, after an undesired behavior occurs. It aims to decrease unwanted behaviors by taking away something the dog values.",
    "03": "Negative reinforcement: This method removes an unpleasant stimulus, such as leash pressure or a verbal correction, when the dog performs the desired behavior. It increases the likelihood of the behavior being repeated to avoid the aversive condition.",
    "04": "Positive punishment: This method introduces an unpleasant stimulus, like a leash jerk, noise, or physical correction, after an unwanted behavior. The intent is to reduce that behavior by adding a negative consequence."
}

# --- Tunable parameters ---
CLUSTER_THRESHOLD = 0.6
TOPN_PER_CLUSTER = 5
KEYBERT_TOPN = 800
VECT_MAX_FEATURES = 10000
FILTER_NUMERIC_TOKEN = True
MIN_DOM_SHARE = 0.5

# Prompt gating
PROMPT_FLOOR = 0.45
PROMPT_GATE_POW = 2.0
PROMPT_GATE_MIN = 0.30


def run_topic_method(conn_str: str):
    """Wrapper to run method keyword extraction and JSD scoring"""
    run_keyword_extraction_with_embeddings(conn_str)
    recompute_jsd_and_update(conn_str)
    
    
# --- Helpers ---
def cluster_keywords(keywords, model, threshold=CLUSTER_THRESHOLD, topn=TOPN_PER_CLUSTER):
    if not keywords:
        return []
    texts = [kw for kw, _ in keywords]
    embeddings = model.encode(texts)
    similarity_matrix = cosine_similarity(embeddings)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1-threshold,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit(1 - similarity_matrix).labels_
    clustered = {}
    for i, label in enumerate(labels):
        if label not in clustered:
            clustered[label] = []
        clustered[label].append((texts[i], keywords[i][1]))
    clustered_topn = []
    for kws in clustered.values():
        kws_sorted = sorted(kws, key=lambda x: x[1], reverse=True)[:topn]
        clustered_topn.extend(kws_sorted)
    return clustered_topn

def cosine_similarity_phrase_to_doc(phrase, doc_text, model):
    doc_emb = model.encode(doc_text, convert_to_tensor=True)
    phrase_emb = model.encode(phrase, convert_to_tensor=True)
    return util.cos_sim(phrase_emb, doc_emb).item()

def get_semantic_relevance(keyword: str, reference_text: str, model) -> float:
    kw_emb = model.encode(keyword, convert_to_tensor=True)
    ref_emb = model.encode(reference_text, convert_to_tensor=True)
    return util.cos_sim(kw_emb, ref_emb).item()

# --- Main extraction ---
def run_keyword_extraction_with_embeddings(conn_str: str):
    start_time = time.time()
    engine = create_engine(conn_str)
    conn = engine.raw_connection()
    cursor = conn.cursor()

    model = SentenceTransformer("all-mpnet-base-v2")
    kw_model = KeyBERT(model)
    all_keywords = defaultdict(list)

    method_text_blobs = {}
    for method_code in ['01', '02', '03', '04']:
        df = pd.read_sql(f"""
            SELECT b.id, w.clean_text
            FROM (
                SELECT DISTINCT id
                FROM dbo.fn_get_baseline_combination('{method_code}', NULL, NULL, NULL)
            ) b
            LEFT JOIN t_web_cleaned w ON b.id = w.response_id
            WHERE w.clean_text IS NOT NULL AND w.clean_text <> ''
        """, conn)

        text_blob = " ".join(df['clean_text'].dropna().tolist())
        method_text_blobs[method_code] = text_blob

        vectorizer = CountVectorizer(
            ngram_range=(1, 3),
            max_features=VECT_MAX_FEATURES,
            stop_words="english"
        ).fit([text_blob])

        keywords = kw_model.extract_keywords(
            text_blob,
            keyphrase_ngram_range=(1, 3),
            vectorizer=vectorizer,
            stop_words="english",
            top_n=KEYBERT_TOPN
        )

        clustered = cluster_keywords(keywords, model)
        for keyword, score in clustered:
            if FILTER_NUMERIC_TOKEN and any(char.isdigit() for char in keyword):
                continue
            sem_score = cosine_similarity_phrase_to_doc(keyword, text_blob, model)
            sem_prompt = get_semantic_relevance(keyword, method_semantic_defs[method_code], model)
            combined_assign_score = 0.5 * sem_score + 0.5 * sem_prompt
            all_keywords[method_code].append((keyword, score, sem_score, sem_prompt, combined_assign_score))

    # Assign exclusivity by dominance
    keyword_best_method = {}
    for method_code, kws in all_keywords.items():
        for kw, score, sem_score, sem_prompt, comb_score in kws:
            if kw not in keyword_best_method or comb_score > keyword_best_method[kw][1]:
                keyword_best_method[kw] = (method_code, comb_score)

    filtered_keywords = defaultdict(list)
    for method_code, kws in all_keywords.items():
        for kw, score, sem_score, sem_prompt, comb_score in kws:
            if keyword_best_method[kw][0] == method_code:
                filtered_keywords[method_code].append((kw, score, sem_score, sem_prompt))

    # Save to DB
    cursor.execute("DELETE FROM t_keywords_by_methods")
    for method_code, kws in filtered_keywords.items():
        for keyword, score, sem_score, sem_prompt in kws:
            cursor.execute("""
                INSERT INTO t_keywords_by_methods
                (method_code, keyword, score, semantic_score, semantic_prompt_score, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, method_code, keyword, score, sem_score, sem_prompt, datetime.datetime.now())

    conn.commit()
    cursor.close()
    conn.close()
    print(f" Keyword extraction done in {time.time() - start_time:.2f} seconds.")

# --- Recompute JSD + final score ---
def recompute_jsd_and_update(
    conn_str: str,
    w_sem_doc: float = 0.30,
    w_jsd: float = 0.20,
    w_prompt: float = 0.50,
    prompt_floor: float = PROMPT_FLOOR,
    prompt_gate_pow: float = PROMPT_GATE_POW,
    prompt_gate_min: float = PROMPT_GATE_MIN
):
    engine = create_engine(conn_str)
    conn = engine.raw_connection()
    cursor = conn.cursor()

    df_keywords = pd.read_sql("""
        SELECT method_code, keyword, semantic_score, semantic_prompt_score
        FROM t_keywords_by_methods
    """, engine).fillna(0.0)

    method_codes = df_keywords['method_code'].unique().tolist()

    # Build text blobs for JSD
    text_blobs = {}
    for code in method_codes:
        df = pd.read_sql(f"""
            SELECT w.clean_text
            FROM (
                SELECT DISTINCT id
                FROM dbo.fn_get_baseline_combination('{code}', NULL, NULL, NULL)
            ) b
            LEFT JOIN t_web_cleaned w ON b.id = w.response_id
            WHERE w.clean_text IS NOT NULL AND w.clean_text <> ''
        """, engine)
        text_blobs[code] = " ".join(df['clean_text'].dropna().tolist()).lower()

    # Compute JSD
    jsd_result = {}
    for kw in df_keywords['keyword'].unique():
        freq = np.array([text_blobs[c].count(kw.lower()) for c in sorted(method_codes)], dtype=float)
        prob = freq / freq.sum() if freq.sum() > 0 else np.full(len(freq), 1e-10)
        uniform = np.full_like(prob, 1 / len(prob))
        jsd = 1 - jensenshannon(prob, uniform, base=2)
        jsd_result[kw] = float(jsd)

    # Final scoring with prompt gating
    def _clip01(x): return float(max(0.0, min(1.0, x)))

    for _, row in df_keywords.iterrows():
        kw = row['keyword']
        sem_doc = _clip01(row['semantic_score'])
        jsd_score = _clip01(jsd_result.get(kw, 0.0))
        sp = _clip01(row['semantic_prompt_score'])

        raw_score = (w_sem_doc * sem_doc) + (w_jsd * jsd_score) + (w_prompt * sp)

        if sp <= prompt_floor:
            gate = 0.0
        else:
            gate = (sp - prompt_floor) / (1.0 - prompt_floor)
        gate = _clip01(gate) ** prompt_gate_pow
        factor = prompt_gate_min + (1.0 - prompt_gate_min) * gate

        final_score = raw_score * factor

        cursor.execute("""
            UPDATE t_keywords_by_methods
            SET jsd_score = ?, final_score = ?
            WHERE method_code = ? AND keyword = ?
        """, jsd_score, final_score, row['method_code'], kw)

    conn.commit()
    cursor.close()
    conn.close()
    print(" Finished updating final scores (prompt-gated).")

# --- Run ---
if __name__ == "__main__":
    conn_str = "mssql+pyodbc://@localhost\\SQLSERV2019/NLP_DOGSTRUST_DEPLOY?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    run_topic_method(conn_str)
