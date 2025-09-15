# run_topic_gender_kl.py

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
from keyword_scoring import recompute_jsd_and_update

# Logging
logger = logging.getLogger("gender_extractor")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh = logging.FileHandler("keyword_gender.log", mode="a", encoding="utf-8")
fh.setFormatter(fmt); fh.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt); ch.setLevel(logging.INFO)
if not logger.handlers:
    logger.addHandler(fh); logger.addHandler(ch)
# ====================================================
# PSEUDOCODE - GENDER-BASED KEYWORD EXTRACTION PIPELINE
# ----------------------------------------------------
# 1. Load gender metadata from DB.
# 2. Fetch all response texts per gender (blob).
# 3. Generate semantic prompts for each gender.
# 4. Load sentence-transformer model & KeyBERT.
# 5. For each gender:
#    a. Extract keywords from blob using KeyBERT.
#    b. Cluster keywords by semantic similarity.
#    c. Filter existing and numeric keywords.
#    d. Compute:
#       - similarity to blob (semantic_doc)
#       - similarity to prompt (semantic_prompt)
#       - average score (combined_assign)
# 6. Assign keyword to the best-matching gender.
# 7. Save final keywords to DB table `t_keywords_by_gender`.
# 8. Ensure required columns exist in the table.
# 9. Call `recompute_jsd_and_update()` to calculate:
#    - JSD distinctiveness
#    - final gated scores
# ====================================================

# Parameters
CLUSTER_THRESHOLD = 0.6
KEYBERT_TOPN = 1200
VECT_MAX_FEATURES = 12000
FILTER_NUMERIC = True

def _norm(s: str) -> str:
    """
    Normalize a string by lowercasing, stripping whitespace, and collapsing spaces.

    Parameters:
        s (str): Input string.

    Returns:
        str: Normalized string.
    """

    return " ".join(s.strip().lower().split())

def _ensure_gender_table_columns(engine):
    """
    Ensures the target SQL table `t_keywords_by_gender` contains required float columns
    for semantic scoring and JSD-based analysis.

    Parameters:
        engine (sqlalchemy.engine.Engine): SQLAlchemy engine connected to target DB.

    Returns:
        None. Alters table in-place if needed.
    """

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



def cluster_keywords(keywords, model, threshold=CLUSTER_THRESHOLD):
    """
    Clusters similar keywords based on semantic similarity using embeddings.

    Parameters:
        keywords (List[Tuple[str, float]]): Keyword and relevance score pairs.
        model (SentenceTransformer): Embedding model.
        threshold (float): Clustering distance threshold (default: 0.6).

    Returns:
        List[Tuple[str, float]]: One top keyword per cluster based on score.
    """

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
    """
    Main pipeline to extract and score gender-specific keywords:

    Steps:
        1. Loads all cleaned text by gender from database.
        2. Builds gender-specific semantic prompts.
        3. Extracts keywords using KeyBERT for each gender blob.
        4. Filters and clusters keywords.
        5. Scores them semantically (to blob + prompt).
        6. Assigns best-matching gender per keyword.
        7. Saves final keywords to DB.
        8. Triggers JSD scoring update.

    Parameters:
        conn_str (str): SQLAlchemy database connection string.

    Returns:
        None. Writes results to `t_keywords_by_gender` table.
    """

    t0 = time.time()
    logger.info("=== GENDER keyword extraction started ===")
    engine = create_engine(conn_str)

    df_gender = pd.read_sql("SELECT gender_code, gender_name FROM m_gender", engine)
    genders = [(str(r.gender_code).zfill(2), r.gender_name) for _, r in df_gender.iterrows()]
    gender_prompts = build_gender_prompts(genders)

    df_existing = pd.read_sql("""
        SELECT keyword FROM t_keywords_by_methods
        UNION
        SELECT keyword FROM t_keywords_by_role
    """, engine)
    existing_keywords = set(_norm(k) for k in df_existing["keyword"].dropna().tolist())

    logger.info("Loading model...")
    model = SentenceTransformer("all-mpnet-base-v2")
    kw_model = KeyBERT(model)

    blobs = {}
    for gc, _ in genders:
        df = pd.read_sql(f"""
            SELECT w.clean_text
            FROM (
                SELECT DISTINCT id
                FROM dbo.fn_get_baseline_combination(NULL, '{gc}', NULL, NULL)
            ) b
            LEFT JOIN t_web_cleaned w ON b.id = w.response_id
            WHERE w.clean_text IS NOT NULL AND w.clean_text <> ''
        """, engine)
        blob = " ".join(df["clean_text"].dropna().tolist())
        blobs[gc] = blob
        logger.info(f"Gender {gc}: rows={len(df)}, blob_len={len(blob)}")

    prompt_embeds = {gc: model.encode(prompt, convert_to_tensor=True) for gc, prompt in gender_prompts.items()}
    blob_embeds = {gc: model.encode(blobs[gc], convert_to_tensor=True) for gc in blobs}

    all_keywords = defaultdict(list)
    for gc in blobs:
        blob = blobs[gc]
        vectorizer = CountVectorizer(
            ngram_range=(1, 3),
            max_features=VECT_MAX_FEATURES,
            stop_words="english"
        ).fit([blob])
        keywords = kw_model.extract_keywords(
            blob,
            keyphrase_ngram_range=(1, 3),
            vectorizer=vectorizer,
            stop_words="english",
            top_n=KEYBERT_TOPN
        )
        clustered = cluster_keywords(keywords, model)
        for kw, kb in clustered:
            norm_kw = _norm(kw)
            if norm_kw in existing_keywords:
                continue
            if FILTER_NUMERIC and any(ch.isdigit() for ch in norm_kw):
                continue
            sem_score = util.cos_sim(model.encode(kw, convert_to_tensor=True), blob_embeds[gc]).item()
            sem_prompt = util.cos_sim(model.encode(kw, convert_to_tensor=True), prompt_embeds[gc]).item()
            ca = 0.5 * float(sem_score) + 0.5 * float(sem_prompt)
            all_keywords[gc].append((kw, float(kb), float(sem_score), float(sem_prompt), float(ca)))

    kw_best = {}
    for gc, kws in all_keywords.items():
        for kw, kb, sd, sp, ca in kws:
            k = _norm(kw)
            if k not in kw_best or ca > kw_best[k][1]:
                kw_best[k] = (gc, ca, kw, kb, sd, sp)

    filtered_keywords = defaultdict(list)
    for _, (gc, ca, kw, kb, sd, sp) in kw_best.items():
        filtered_keywords[gc].append((kw, kb, sd, sp))

    _ensure_gender_table_columns(engine)

    with engine.begin() as cn:
        cn.exec_driver_sql("DELETE FROM t_keywords_by_gender")
        for gc, items in filtered_keywords.items():
            for kw, kb, sd, sp in items:
                cn.exec_driver_sql("""
                    INSERT INTO t_keywords_by_gender
                    (gender_code, keyword, score, semantic_score, semantic_prompt_score, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (gc, kw, kb, sd, sp, datetime.datetime.now()))

    recompute_jsd_and_update(
        conn_str=conn_str,
        table_name="t_keywords_by_gender",
        category_col="gender_code",
        baseline_func_col="gender_code"
    )

    logger.info(f"=== GENDER keyword extraction done in {time.time()-t0:.2f}s ===")
    
    def build_gender_prompts(genders):
        """
    Builds predefined semantic prompts for each gender based on gender_code.

    Parameters:
        genders (List[Tuple[str, str]]): List of (gender_code, gender_name) tuples.

    Returns:
        Dict[str, str]: Mapping of gender_code to semantic prompt text.
    """

        out = {}
    for gc, gn in genders:
        gc_str = str(gc).zfill(2)
        if gc_str == "01":
            out[gc] = "This webpage reflects the language of a female canine professional. Look for gendered terms such as she, her, female, etc."
        elif gc_str == "02":
            out[gc] = "This webpage reflects the language of a male canine professional. Look for gendered terms such as he, him, male, etc."
        elif gc_str == "03":
            out[gc] = "This webpage reflects the language of a non-binary canine professional. Look for inclusive terms such as they, their, etc."
        else:
            out[gc] = f"This webpage reflects language associated with {gn or f'gender {gc}'} canine professionals."
    return out

if __name__ == "__main__":
    conn_str = r"mssql+pyodbc://@localhost\SQLSERV2019/NLP_DOGSTRUST_DEPLOY?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    run_topic_gender(conn_str)
