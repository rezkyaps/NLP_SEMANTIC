# file: run_topic_gender_optimized.py
import time
import logging
import sys
from typing import Dict, List, Tuple
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import datetime
from sqlalchemy import create_engine, text
from collections import defaultdict
from scipy.spatial.distance import jensenshannon

# ========= Logging =========
logger = logging.getLogger("gender_extractor")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh = logging.FileHandler("keyword_gender.log", mode="a", encoding="utf-8")
fh.setFormatter(fmt); fh.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt); ch.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(fh); logger.addHandler(ch)

# ========= Params =========
CLUSTER_THRESHOLD = 0.6
KEYBERT_TOPN = 1200
VECT_MAX_FEATURES = 12000
FILTER_NUMERIC = True

W_SEM_DOC = 0.30
W_JSD     = 0.40
W_PROMPT  = 0.30

def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())

def detect_gender_table(engine):
    with engine.connect() as c:
        names = pd.read_sql("SELECT name FROM sys.tables", c)["name"].str.lower().tolist()
    if "m_gender" in names:
        return "m_gender", "gender_code", "gender_name"
    if "m_genders" in names:
        return "m_genders", "gender_code", "gender_name"
    raise RuntimeError("Gender master table not found")

def _ensure_gender_table_columns(engine):
    with engine.begin() as cn:
        cn.exec_driver_sql("""
            IF COL_LENGTH('dbo.t_keywords_by_gender','semantic_score') IS NULL
                ALTER TABLE dbo.t_keywords_by_gender ADD semantic_score FLOAT NULL;
            IF COL_LENGTH('dbo.t_keywords_by_gender','semantic_prompt_score') IS NULL
                ALTER TABLE dbo.t_keywords_by_gender ADD semantic_prompt_score FLOAT NULL;
            IF COL_LENGTH('dbo.t_keywords_by_gender','jsd_score') IS NULL
                ALTER TABLE dbo.t_keywords_by_gender ADD jsd_score FLOAT NULL;
            IF COL_LENGTH('dbo.t_keywords_by_gender','final_score') IS NULL
                ALTER TABLE dbo.t_keywords_by_gender ADD final_score FLOAT NULL;
            IF COL_LENGTH('dbo.t_keywords_by_gender','score') IS NULL
                ALTER TABLE dbo.t_keywords_by_gender ADD score FLOAT NULL;
        """)

def build_gender_prompts(genders):
    out = {}
    for gc, gn in genders:
        gc_str = str(gc).zfill(2)
        if gc_str == "01":
            out[gc] = ("This webpage reflects the language of a female canine professional. "
                       "Look for gendered terms such as: she, her, hers, woman, female, mum, lady, girls, daughter. "
                       "Language may include nurturing, empathy, patience, softness, and caregiving tone. "
                       "Avoid generic words unrelated to gender identity.")
        elif gc_str == "02":
            out[gc] = ("This webpage reflects the language of a male canine professional. "
                       "Look for gendered terms such as: he, him, his, man, male, dad, gentleman, boys, son. "
                       "Language may include directness, authority, firmness, or confidence. "
                       "Avoid terms unrelated to personal identity or communication style.")
        elif gc_str == "03":
            out[gc] = ("This webpage reflects a non-binary or gender-diverse canine professional. "
                       "Look for inclusive or neutral terms such as: they, them, their, Mx, partner, parent, caregiver, client, practitioner. "
                       "Language may avoid gendered pronouns or use identity-neutral phrasing. "
                       "Avoid stereotypical gendered descriptors.")
        else:
            out[gc] = f"This webpage reflects language associated with {gn or f'gender {gc}'} canine professionals."
    return out

def cluster_keywords_representative(keywords, model, threshold=CLUSTER_THRESHOLD):
    if not keywords:
        return []
    texts = [kw for kw, _ in keywords]
    if len(texts) == 1:
        return [(texts[0], keywords[0][1])]
    emb = model.encode(texts)
    sim = cosine_similarity(emb)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - threshold,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit(1 - sim).labels_
    best = {}
    for i, lab in enumerate(labels):
        cand = (texts[i], keywords[i][1])
        if lab not in best or cand[1] > best[lab][1]:
            best[lab] = cand
    return list(best.values())

def run_topic_gender(conn_str: str):
    t0 = time.time()
    logger.info("=== GENDER keyword extraction started ===")
    engine = create_engine(conn_str)

    # Detect table
    gender_table, gender_code_col, gender_name_col = detect_gender_table(engine)

    # Exclude anything already taken by method/role
    df_existing = pd.read_sql("""
        SELECT keyword FROM t_keywords_by_methods
        UNION
        SELECT keyword FROM t_keywords_by_role
    """, engine)
    existing_keywords = set(_norm(k) for k in df_existing["keyword"].dropna().tolist())

    # Load genders
    df_gender = pd.read_sql(f"SELECT {gender_code_col}, {gender_name_col} FROM {gender_table}", engine)
    genders = [(str(r[gender_code_col]).zfill(2), r[gender_name_col]) for _, r in df_gender.iterrows()]
    if not genders:
        logger.warning("No genders found. Exiting.")
        return

    # Prompts + explicit pronoun signals
    gender_prompts = build_gender_prompts(genders)
    pronoun_signals = {
        "01": ["she", "her", "hers", "woman", "female", "mum", "lady", "girls", "daughter"],
        "02": ["he", "him", "his", "man", "male", "dad", "gentleman", "boys", "son"],
        "03": ["they", "them", "their", "theirs", "mx", "parent", "caregiver", "partner", "client", "practitioner"]
    }

    # Model
    logger.info("Loading sentence-transformer model…")
    model = SentenceTransformer("all-mpnet-base-v2")
    kw_model = KeyBERT(model)

    # Build blobs (once)
    gender_text_blobs: Dict[str, str] = {}
    with engine.connect() as c:
        for gc, _ in genders:
            df = pd.read_sql(f"""
                SELECT w.clean_text
                FROM (
                    SELECT DISTINCT id
                    FROM dbo.fn_get_baseline_combination(NULL, '{gc}', NULL, NULL)
                ) b
                LEFT JOIN t_web_cleaned w ON b.id = w.response_id
                WHERE w.clean_text IS NOT NULL AND w.clean_text <> ''
            """, c)
            blob = " ".join(df["clean_text"].dropna().tolist())
            gender_text_blobs[gc] = blob
            logger.info(f"Gender {gc}: rows={len(df)}, blob_len={len(blob)}")

    # Pre-encode blob & prompt per gender
    blob_embeds = {}
    prompt_embeds = {}
    for gc, _ in genders:
        blob_text = gender_text_blobs.get(gc, "")
        blob_embeds[gc] = model.encode(blob_text, convert_to_tensor=True)
        prompt_text = gender_prompts.get(gc, "")
        prompt_embeds[gc] = model.encode(prompt_text, convert_to_tensor=True) if prompt_text else None

    # Extract & score per gender
    all_keywords = defaultdict(list)  # gc -> list[(kw, kb, sd, sp, ca)]
    for gc, _ in genders:
        text_blob = gender_text_blobs.get(gc, "")
        if not text_blob.strip():
            continue

        # Seed with explicit signals if present
        blob_lc = text_blob.lower()
        for sig in pronoun_signals.get(gc, []):
            if sig in blob_lc and _norm(sig) not in existing_keywords:
                sd = util.cos_sim(model.encode(sig, convert_to_tensor=True), blob_embeds[gc]).item()
                sp = util.cos_sim(model.encode(sig, convert_to_tensor=True), prompt_embeds[gc]).item() if prompt_embeds[gc] is not None else 0.0
                ca = 0.5 * float(sd) + 0.5 * float(sp)
                all_keywords[gc].append((sig, 0.95, float(sd), float(sp), float(ca)))

        # KeyBERT
        try:
            vectorizer = CountVectorizer(
                ngram_range=(1, 3),
                max_features=VECT_MAX_FEATURES,
                stop_words="english"
            ).fit([text_blob])
            raw_keywords = kw_model.extract_keywords(
                text_blob,
                keyphrase_ngram_range=(1, 3),
                vectorizer=vectorizer,
                stop_words="english",
                top_n=KEYBERT_TOPN
            )
        except Exception:
            logger.exception(f"KeyBERT failed for gender {gc}")
            continue

        # Dedup keep-best
        best_by_text = {}
        for k, s in raw_keywords:
            if k not in best_by_text or s > best_by_text[k]:
                best_by_text[k] = s
        dedup = [(k, best_by_text[k]) for k in best_by_text]

        clustered = cluster_keywords_representative(dedup, model)
        if not clustered:
            continue

        kw_texts = [kw for kw, _ in clustered]
        kw_emb = model.encode(kw_texts, convert_to_tensor=True, batch_size=128, show_progress_bar=False)

        # Vectorised sims
        sd_vec = util.cos_sim(kw_emb, blob_embeds[gc]).cpu().numpy().reshape(-1)
        if prompt_embeds[gc] is not None:
            sp_vec = util.cos_sim(kw_emb, prompt_embeds[gc]).cpu().numpy().reshape(-1)
        else:
            sp_vec = np.zeros(len(kw_texts), dtype=float)

        for (kw, kb), sd, sp in zip(clustered, sd_vec, sp_vec):
            k_norm = _norm(kw)
            if FILTER_NUMERIC and any(ch.isdigit() for ch in k_norm):
                continue
            if k_norm in existing_keywords:
                continue
            ca = 0.5 * float(sd) + 0.5 * float(sp)
            all_keywords[gc].append((kw, float(kb), float(sd), float(sp), float(ca)))

    # Inter-gender exclusivity by highest ca
    kw_best_gender = {}
    for gc, kws in all_keywords.items():
        for kw, kb, sd, sp, ca in kws:
            k = _norm(kw)
            if k not in kw_best_gender or ca > kw_best_gender[k][1]:
                kw_best_gender[k] = (gc, ca, kw, kb, sd, sp)

    filtered_keywords = defaultdict(list)  # gc -> list[(kw, kb, sd, sp)]
    for _, (gc, ca, kw, kb, sd, sp) in kw_best_gender.items():
        filtered_keywords[gc].append((kw, kb, sd, sp))

    # Ensure columns exist
    _ensure_gender_table_columns(engine)

    # Build lowercased blobs for JSD
    gc_sorted = sorted(filtered_keywords)
    blobs_lower = {gc: gender_text_blobs.get(gc, "").lower() for gc in gc_sorted}

    # JSD map (stabilised; 1 - JSD(prob || uniform))
    jsd_map = {}
    for k_norm in kw_best_gender.keys():
        freq = np.array([blobs_lower[gc].count(k_norm) for gc in gc_sorted], dtype=float)
        if freq.sum() <= 0:
            prob = np.full(len(freq), 1.0 / max(1, len(freq)), dtype=float)
        else:
            prob = freq / freq.sum()
        eps = 1e-12
        prob = np.clip(prob, eps, 1.0); prob = prob / prob.sum()
        uniform = np.full_like(prob, 1.0 / len(prob))
        uniform = np.clip(uniform, eps, 1.0); uniform = uniform / uniform.sum()
        d = jensenshannon(prob, uniform, base=2)
        if not np.isfinite(d):
            d = 0.0
        jsd01 = float(1.0 - d)
        if not np.isfinite(jsd01):
            jsd01 = 0.0
        jsd_map[k_norm] = jsd01

    # Save (replace)
    with engine.begin() as cn:
        cn.exec_driver_sql("DELETE FROM t_keywords_by_gender")
        for gc, items in filtered_keywords.items():
            for kw, kb, sd, sp in items:
                k_norm = _norm(kw)
                jsd_score = float(jsd_map.get(k_norm, 0.0))
                # Clamp to finite
                def clip01(x):
                    x = float(x); 
                    if not np.isfinite(x): return 0.0
                    return float(max(0.0, min(1.0, x)))
                kb = 0.0 if not np.isfinite(kb) else float(kb)
                sd = clip01(sd)
                sp = clip01(sp)
                jsd_score = clip01(jsd_score)
                final_score = float(W_SEM_DOC*sd + W_JSD*jsd_score + W_PROMPT*sp)
                cn.exec_driver_sql("""
                    INSERT INTO t_keywords_by_gender
                    (gender_code, keyword, score, semantic_score, semantic_prompt_score, jsd_score, final_score, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (gc, kw, kb, sd, sp, jsd_score, final_score, datetime.datetime.now()))

    logger.info(f"=== GENDER keyword extraction done in {time.time()-t0:.2f}s ===")

if __name__ == "__main__":
    conn_str = r"mssql+pyodbc://@localhost\SQLSERV2019/NLP_DOGSTRUST_DEPLOY?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    run_topic_gender(conn_str)
