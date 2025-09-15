# file: merge.py

import pandas as pd
import numpy as np
import datetime
import logging
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer

logging.basicConfig(
    filename="semantic_stratified_merge.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def get_top_clusters(engine, top_n=5):
    logging.info("Fetching top clusters from database...")
    query = '''
        SELECT method_code, gender_code, role_code, COUNT(DISTINCT id) AS total_respondents
        FROM dbo.fn_get_baseline_combination(NULL, NULL, NULL, NULL)
        GROUP BY method_code, gender_code, role_code
        ORDER BY total_respondents DESC
    '''
    df = pd.read_sql(query, engine)
    df['cluster_code'] = df['method_code'] + '-' + df['gender_code'] + '-' + df['role_code']
    return df.head(top_n), df

def fetch_responden_texts(engine):
    logging.info("Fetching responden texts for clustering...")
    query = '''
        SELECT DISTINCT b.id, w.clean_text, b.method_code, b.gender_code, b.role_code,
               CONCAT(b.method_code, '-', b.gender_code, '-', b.role_code) AS cluster_code
        FROM dbo.fn_get_baseline_combination(NULL, NULL, NULL, NULL) b
        JOIN t_web_cleaned w ON b.id = w.response_id
        WHERE w.clean_text IS NOT NULL AND w.clean_text <> ''
    '''
    return pd.read_sql(query, engine)

def compute_cluster_embeddings(df, model):
    logging.info("Computing base cluster embeddings...")
    cluster_embeds = {}
    for cluster_code, group in df.groupby("cluster_code"):
        texts = group["clean_text"].tolist()
        embeds = model.encode(texts)
        cluster_embeds[cluster_code] = np.mean(embeds, axis=0)
    return cluster_embeds

def score_similarity(base_cluster_vecs, target_vec, row, alpha=0.7, beta=0.3):
    best_score = -1
    best_cluster = None
    for code, base_vec in base_cluster_vecs.items():
        text_sim = util.cos_sim(target_vec, base_vec).item()
        feat_sim = int(code.split('-')[0] == row['method_code']) + \
                   int(code.split('-')[1] == row['gender_code']) + \
                   int(code.split('-')[2] == row['role_code'])
        feat_sim /= 3
        score = alpha * text_sim + beta * feat_sim
        if score > best_score:
            best_score = score
            best_cluster = code
    return best_cluster, best_score

def assign_clusters(df, base_codes, base_cluster_vecs, model):
    logging.info("Assigning clusters based on similarity...")
    assignments = []
    for _, row in df.iterrows():
        target_vec = model.encode(row['clean_text'], convert_to_tensor=True)
        best_cluster, best_score = score_similarity(base_cluster_vecs, target_vec, row)
        assignments.append((row['id'], best_cluster, best_score))
    return pd.DataFrame(assignments, columns=['id', 'final_cluster', 'similarity_score'])

def extract_keywords(engine, df_final, model):
    logging.info("Extracting and saving keywords per cluster...")
    kw_model = KeyBERT(model)
    all_keywords = defaultdict(list)
    cursor = engine.raw_connection().cursor()
    cursor.execute("DELETE FROM t_keywords_by_cluster")
    for cluster_code, group in df_final.groupby("final_cluster"):
        texts = group["clean_text"].dropna().unique().tolist()
        blob = " ".join(texts)
        if not blob.strip():
            continue
        vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words="english", max_features=4000)
        keywords = kw_model.extract_keywords(blob, vectorizer=vectorizer, top_n=100, stop_words="english")
        for kw, score in keywords:
            sem_score = util.cos_sim(model.encode(kw, convert_to_tensor=True), model.encode(blob, convert_to_tensor=True)).item()
            all_keywords[cluster_code].append((kw, score, sem_score))
            cursor.execute("""
                INSERT INTO t_keywords_by_cluster (cluster_id, keyword, score, semantic_score, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, cluster_code, kw, score, sem_score, datetime.datetime.now())
    cursor.connection.commit()
    cursor.close()

def merge_data(conn_str):
    logging.info("[MERGE] Starting semantic merge process...")
    engine = create_engine(conn_str)
    model = SentenceTransformer("all-mpnet-base-v2")

    top5_df, full_df = get_top_clusters(engine, top_n=5)
    base_codes = top5_df["cluster_code"].tolist()
    df_all = fetch_responden_texts(engine)

    df_all = df_all[df_all["cluster_code"].isin(full_df["cluster_code"])]
    base_df = df_all[df_all["cluster_code"].isin(base_codes)]
    rest_df = df_all[~df_all["cluster_code"].isin(base_codes)]

    base_cluster_vecs = compute_cluster_embeddings(base_df, model)
    assigned_df = assign_clusters(rest_df, base_codes, base_cluster_vecs, model)

    final_df = pd.merge(df_all, assigned_df, on="id", how="left")
    final_df["final_cluster"] = final_df["final_cluster"].fillna(final_df["cluster_code"])
    extract_keywords(engine, final_df, model)

    final_df[["id", "final_cluster"]].to_sql("responden_cluster_map", engine, if_exists="replace", index=False)
    logging.info("[MERGE] Merge completed and stored to SQL.")
