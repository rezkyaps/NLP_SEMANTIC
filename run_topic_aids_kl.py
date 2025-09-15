import time
import logging
import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
from sqlalchemy import create_engine

# Logging Setup
logger = logging.getLogger("aid_extractor")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh = logging.FileHandler("keyword_aidtype.log", mode="a", encoding="utf-8")
fh.setFormatter(formatter)
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(formatter)
ch.setLevel(logging.INFO)
if not logger.hasHandlers():
    logger.addHandler(fh)
    logger.addHandler(ch)

# Constants
CLUSTER_THRESHOLD = 0.6
KEYBERT_TOPN = 1000
VECT_MAX_FEATURES = 10000
W_SEM_DOC = 0.30
W_JSD = 0.40
W_PROMPT = 0.30
FILTER_NUMERIC = True


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def cluster_keywords(keywords, model, threshold=CLUSTER_THRESHOLD):
    if not keywords:
        return []
    texts = [kw for kw, _ in keywords]
    embeddings = model.encode(texts)
    sim_matrix = cosine_similarity(embeddings)
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1 - threshold, metric='precomputed', linkage='average')
    labels = clustering.fit(1 - sim_matrix).labels_
    clustered = {}
    for i, label in enumerate(labels):
        if label not in clustered or keywords[i][1] > clustered[label][1]:
            clustered[label] = (texts[i], keywords[i][1])
    return list(clustered.values())


def run_topic_aids(conn_str: str):
    start = time.time()
    logger.info(" START: Keyword extraction by aid type + code")

    engine = create_engine(conn_str)
    model = SentenceTransformer("all-mpnet-base-v2")
    kw_model = KeyBERT(model)

    logger.info("📥 Loading existing keywords and aid list...")
    with engine.connect() as conn:
        df_methods = pd.read_sql("SELECT keyword FROM t_keywords_by_methods", conn)
        df_roles = pd.read_sql("SELECT keyword FROM t_keywords_by_role", conn)
        df_gender = pd.read_sql("SELECT keyword FROM t_keywords_by_gender", conn)
        existing_keywords = set(
            _norm(k)
            for k in pd.concat([df_methods, df_roles, df_gender])["keyword"].dropna().tolist()
        )
        df_aid = pd.read_sql("SELECT DISTINCT aid_type_code, aid_code, aid_name FROM m_aids_used WHERE aid_name IS NOT NULL", conn)

    all_keywords = defaultdict(list)
    aid_blobs_lower = defaultdict(str)

    logger.info(" Starting KeyBERT extraction and clustering...")
    with engine.connect() as conn:
        for _, row in df_aid.iterrows():
            aid_type = row["aid_type_code"]
            aid_code = row["aid_code"]
            aid_name = row["aid_name"].replace("-", " ") if row["aid_name"] else ""

            prompt = (
                f"Please extract keywords that are tools or synonyms of tools from this text. "
                f"This text is about the specific tool '{aid_name}' in the category '{aid_type}'. "
                f"Your task is to identify domain-specific aids, equipment names, product-related tools, or synonyms that relate to this particular tool. "
                f"Avoid abstract, vague, or non-tool-specific keywords."
            )

            df = pd.read_sql(f"""
                SELECT w.clean_text
                FROM (
                    SELECT DISTINCT id
                    FROM dbo.fn_get_baseline_combination(NULL, NULL, NULL, '{aid_type}')
                    WHERE aid_code = '{aid_code}'
                ) b
                LEFT JOIN t_web_cleaned w ON b.id = w.response_id
                WHERE w.clean_text IS NOT NULL AND w.clean_text <> ''
            """, conn)

            blob = " ".join(df["clean_text"].dropna().tolist())
            if not blob.strip():
                logger.warning(f" Empty text blob for aid_type={aid_type}, aid_code={aid_code}")
                continue

            aid_blobs_lower[(aid_type, aid_code)] = blob.lower()
            logger.info(f"🔍 Processing aid_type={aid_type}, aid_code={aid_code}, rows={len(df)}")

            try:
                vect = CountVectorizer(ngram_range=(1, 3), max_features=VECT_MAX_FEATURES, stop_words="english").fit([blob])
                raw_kws = kw_model.extract_keywords(blob, vectorizer=vect, top_n=KEYBERT_TOPN, stop_words="english")
            except Exception:
                logger.exception(f" KeyBERT failed for aid_type={aid_type}, aid_code={aid_code}")
                continue

            best_text = {k: s for k, s in raw_kws if _norm(k) not in existing_keywords}
            dedup = [(k, best_text[k]) for k in best_text]
            clustered = cluster_keywords(dedup, model)
            if not clustered:
                logger.warning(f" No clustered keywords for aid_type={aid_type}, aid_code={aid_code}")
                continue

            kw_texts = [kw for kw, _ in clustered]
            kw_emb = model.encode(kw_texts, convert_to_tensor=True)
            blob_emb = model.encode(blob, convert_to_tensor=True)
            prompt_emb = model.encode(prompt, convert_to_tensor=True)

            sd_vec = util.cos_sim(kw_emb, blob_emb).cpu().numpy().reshape(-1)
            sp_vec = util.cos_sim(kw_emb, prompt_emb).cpu().numpy().reshape(-1)

            for (kw, kb), sd, sp in zip(clustered, sd_vec, sp_vec):
                k_norm = _norm(kw)
                if FILTER_NUMERIC and any(ch.isdigit() for ch in k_norm):
                    continue
                comb_score = 0.5 * float(sd) + 0.5 * float(sp)
                all_keywords[(aid_type, aid_code)].append((kw, float(kb), float(sd), float(sp), comb_score))

    logger.info(" Aggregating top keywords and calculating JSD...")
    kw_best = {}
    for (aid_type, aid_code), kws in all_keywords.items():
        for kw, kb, sd, sp, ca in kws:
            k = _norm(kw)
            if (k not in kw_best) or ca > kw_best[k][4]:
                kw_best[k] = (aid_type, aid_code, kw, kb, sd, sp, ca)

    aid_keys = list(set([k[:2] for k in kw_best.values()]))
    all_keys = list(set(k for k in kw_best))
    jsd_map = {}
    for k_norm in all_keys:
        freq = np.array([
            aid_blobs_lower[key].count(k_norm) for key in aid_keys
        ], dtype=float)
        prob = freq / freq.sum() if freq.sum() > 0 else np.full(len(freq), 1.0 / len(freq))
        prob = np.clip(prob, 1e-12, 1.0); prob /= prob.sum()
        uniform = np.full_like(prob, 1.0 / len(prob))
        jsd = jensenshannon(prob, uniform, base=2)
        jsd_map[k_norm] = float(1.0 - jsd if np.isfinite(jsd) else 0.0)

    logger.info(" Force-including aid_name keywords into result with scores...")
    for _, row in df_aid.iterrows():
        aid_type = row["aid_type_code"]
        aid_code = row["aid_code"]
        kw = row["aid_name"].replace("_", " ").replace("-", " ").lower().strip()

        if kw in kw_best:
            continue

        blob = aid_blobs_lower.get((aid_type, aid_code), "")
        blob_embed = model.encode(blob, convert_to_tensor=True)
        prompt = (
            f"Please extract keywords that are tools or synonyms of tools from this text. "
            f"This text is about the specific tool '{kw}' in the category '{aid_type}'. "
            f"Your task is to identify domain-specific aids, equipment names, product-related tools, or synonyms that relate to this particular tool. "
            f"Avoid abstract, vague, or non-tool-specific keywords."
        )
        prompt_embed = model.encode(prompt, convert_to_tensor=True)
        kw_embed = model.encode(kw, convert_to_tensor=True)

        sd = float(util.cos_sim(kw_embed, blob_embed).cpu().numpy().reshape(-1)[0])
        sp = float(util.cos_sim(kw_embed, prompt_embed).cpu().numpy().reshape(-1)[0])
        ca = 0.5 * sd + 0.5 * sp

        kw_best[kw] = (aid_type, aid_code, kw, 0.90, sd, sp, ca)
        logger.info(f" aid_name keyword force-included: {kw} — sd={sd:.4f}, sp={sp:.4f}, ca={ca:.4f}")

    def clip01(x): return float(max(0.0, min(1.0, float(x)))) if np.isfinite(x) else 0.0

    logger.info(" Inserting final keywords into SQL table...")
    with engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM t_keywords_by_aid_type")
        for k_norm, (aid_type, aid_code, kw, kb, sd, sp, ca) in kw_best.items():
            jsd_score = clip01(jsd_map.get(k_norm, 0.0))
            final_score = float(W_SEM_DOC * clip01(sd) + W_JSD * jsd_score + W_PROMPT * clip01(sp))
            conn.exec_driver_sql(
                """
                INSERT INTO t_keywords_by_aid_type
                (aid_type_code, aid_code, keyword, score, semantic_score, semantic_prompt_score, jsd_score, final_score, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (aid_type, aid_code, kw, float(kb), float(sd), float(sp), jsd_score, final_score, datetime.datetime.now())
            )

    logger.info(f" DONE. Inserted {len(kw_best)} keywords in {time.time() - start:.2f}s")


if __name__ == "__main__":
    conn_str = "mssql+pyodbc://@localhost\\SQLSERV2019/NLP_DOGSTRUST_DEPLOY?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    run_topic_aids(conn_str)
