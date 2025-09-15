import time
import logging
import sys
from typing import Dict, List
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
from keyword_scoring import recompute_jsd_and_update  # Reused scoring logic

# ========= Logging =========
logger = logging.getLogger("role_extractor")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh = logging.FileHandler("keyword_role.log", mode="a", encoding="utf-8")
fh.setFormatter(fmt)
fh.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)
ch.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(fh)
    logger.addHandler(ch)

# ========= Parameters =========
CLUSTER_THRESHOLD = 0.6
KEYBERT_TOPN = 1200
VECT_MAX_FEATURES = 12000
FILTER_NUMERIC = True


def _norm(s: str) -> str:
    return " ".join(s.strip().lower().split())


def _ensure_role_table_columns(engine):
    with engine.begin() as cn:
        cn.exec_driver_sql("""
            IF COL_LENGTH('dbo.t_keywords_by_role','semantic_score') IS NULL
                ALTER TABLE dbo.t_keywords_by_role ADD semantic_score FLOAT NULL;
            IF COL_LENGTH('dbo.t_keywords_by_role','semantic_prompt_score') IS NULL
                ALTER TABLE dbo.t_keywords_by_role ADD semantic_prompt_score FLOAT NULL;
            IF COL_LENGTH('dbo.t_keywords_by_role','jsd_score') IS NULL
                ALTER TABLE dbo.t_keywords_by_role ADD jsd_score FLOAT NULL;
            IF COL_LENGTH('dbo.t_keywords_by_role','final_score') IS NULL
                ALTER TABLE dbo.t_keywords_by_role ADD final_score FLOAT NULL;
            IF COL_LENGTH('dbo.t_keywords_by_role','score') IS NULL
                ALTER TABLE dbo.t_keywords_by_role ADD score FLOAT NULL;
        """)


def cluster_keywords(keywords, model, threshold=CLUSTER_THRESHOLD):
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


def build_role_prompts(roles):
    out = {}
    for rc, rn in roles:
        rc_str = str(rc).zfill(2)
        if rc_str == "01":
            out[rc] = ("This webpage reflects an accredited dog trainer. "
                       "Look for terms related to certification, evidence-based methods, official programs, or standard practices."
                       " It may include references to CPD, memberships, qualifications, formal training schools, or credentials.")
        elif rc_str == "02":
            out[rc] = ("This webpage reflects a qualified behaviourist. "
                       "Look for terminology related to behavioural assessment, animal psychology, clinical cases, problem solving, or welfare focus."
                       " Words like behaviour modification, emotional state, diagnosis, or treatment plans may appear.")
        elif rc_str == "03":
            out[rc] = ("This webpage reflects a professional dog trainer without formal qualification. "
                       "It may contain personal experience stories, informal techniques, intuitive language, or anecdotal evidence."
                       " Language may lack reference to formal education or credentials.")
        else:
            out[rc] = f"This webpage reflects the language of role code {rc_str} ({rn or 'unknown role'})."
    return out


def run_topic_role(conn_str: str):
    t0 = time.time()
    logger.info("=== ROLE keyword extraction started ===")
    engine = create_engine(conn_str)

    # Read roles and construct prompts
    df_roles = pd.read_sql("SELECT role_code, role_name FROM m_roles", engine)
    roles = [(str(r.role_code).zfill(2), r.role_name) for _, r in df_roles.iterrows()]
    role_prompts = build_role_prompts(roles)

    # Exclude keywords from method
    df_existing = pd.read_sql("SELECT keyword FROM t_keywords_by_methods", engine)
    existing_keywords = set(_norm(k) for k in df_existing["keyword"].dropna().tolist())

    logger.info("Loading sentence-transformer model…")
    model = SentenceTransformer("all-mpnet-base-v2")
    kw_model = KeyBERT(model)

    blobs = {}
    for rc, _ in roles:
        df = pd.read_sql(f"""
            SELECT w.clean_text
            FROM (
                SELECT DISTINCT id
                FROM dbo.fn_get_baseline_combination(NULL, NULL, '{rc}', NULL)
            ) b
            LEFT JOIN t_web_cleaned w ON b.id = w.response_id
            WHERE w.clean_text IS NOT NULL AND w.clean_text <> ''
        """, engine)
        blob = " ".join(df["clean_text"].dropna().tolist())
        blobs[rc] = blob
        logger.info(f"Role {rc}: rows={len(df)}, blob_len={len(blob)}")

    blob_embeds = {rc: model.encode(text, convert_to_tensor=True) for rc, text in blobs.items()}
    prompt_embeds = {rc: model.encode(prompt, convert_to_tensor=True) for rc, prompt in role_prompts.items()}

    all_keywords = defaultdict(list)
    for rc, _ in roles:
        text_blob = blobs.get(rc, "")
        if not text_blob.strip():
            continue
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
        best_by_text = {k: max([s for k2, s in raw_keywords if k2 == k]) for k, _ in raw_keywords}
        dedup = [(k, best_by_text[k]) for k in best_by_text]
        clustered = cluster_keywords(dedup, model)

        kw_texts = [kw for kw, _ in clustered]
        kw_emb = model.encode(kw_texts, convert_to_tensor=True, batch_size=128, show_progress_bar=False)

        sd_vec = util.cos_sim(kw_emb, blob_embeds[rc]).cpu().numpy().reshape(-1)
        sp_vec = util.cos_sim(kw_emb, prompt_embeds[rc]).cpu().numpy().reshape(-1)

        for (kw, kb), sd, sp in zip(clustered, sd_vec, sp_vec):
            k_norm = _norm(kw)
            if FILTER_NUMERIC and any(ch.isdigit() for ch in k_norm):
                continue
            if k_norm in existing_keywords:
                continue
            ca = 0.5 * float(sd) + 0.5 * float(sp)
            all_keywords[rc].append((kw, float(kb), float(sd), float(sp), float(ca)))

    kw_best = {}
    for rc, kws in all_keywords.items():
        for kw, kb, sd, sp, ca in kws:
            k = _norm(kw)
            if k not in kw_best or ca > kw_best[k][1]:
                kw_best[k] = (rc, ca, kw, kb, sd, sp)

    filtered_keywords = defaultdict(list)
    for _, (rc, ca, kw, kb, sd, sp) in kw_best.items():
        filtered_keywords[rc].append((kw, kb, sd, sp))

    _ensure_role_table_columns(engine)

    with engine.begin() as cn:
        cn.exec_driver_sql("DELETE FROM t_keywords_by_role")
        for rc, items in filtered_keywords.items():
            for kw, kb, sd, sp in items:
                cn.exec_driver_sql("""
                    INSERT INTO t_keywords_by_role
                    (role_code, keyword, score, semantic_score, semantic_prompt_score, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (rc, kw, kb, sd, sp, datetime.datetime.now()))

    # --- Call shared scoring ---
    recompute_jsd_and_update(conn_str, table_name="t_keywords_by_role", code_column="role_code")

    logger.info(f"=== ROLE keyword extraction done in {time.time()-t0:.2f}s ===")


if __name__ == "__main__":
    conn_str = r"mssql+pyodbc://@localhost\SQLSERV2019/NLP_DOGSTRUST_DEPLOY?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    run_topic_role(conn_str)