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
##############################################
# PSEUDOCODE - SEMANTIC STRATIFIED MERGE
##############################################

# 1. Load top-N respondent clusters from SQL (by method, gender, role).
# 2. Load all available respondent texts and their original cluster codes.
# 3. Separate data into:
#     a. Base clusters (top-N most populated)
#     b. Remaining clusters (to be reassigned)
# 4. Compute average sentence embeddings per base cluster.
# 5. For each respondent in the non-base group:
#     a. Encode their text
#     b. Calculate cosine similarity to each base cluster
#     c. Adjust score using feature overlap (method/gender/role)
#     d. Assign to the cluster with highest adjusted similarity
# 6. Merge reassignment with all respondents.
# 7. For each final cluster:
#     a. Concatenate all texts
#     b. Extract top keywords using KeyBERT
#     c. Save keywords + scores (textual + semantic) into SQL
# 8. Save final respondent-to-cluster mapping into SQL table.

def get_top_clusters(engine, top_n=5):
    """
    Fetch the top-N most frequent method-gender-role combinations (clusters).

    Parameters:
        engine (sqlalchemy.Engine): SQLAlchemy engine to query the database.
        top_n (int): Number of top clusters to return.

    Returns:
        tuple:
            - top_df (pd.DataFrame): Top-N clusters with respondent counts.
            - all_df (pd.DataFrame): Full cluster set with counts.
    """

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
    """
    Retrieve cleaned text and cluster metadata for each respondent.

    Parameters:
        engine (sqlalchemy.Engine): SQLAlchemy engine to query the database.

    Returns:
        pd.DataFrame: Respondent data including method, gender, role codes and clean text.
    """

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
    """
    Compute average embedding vectors for each base cluster.

    Parameters:
        df (pd.DataFrame): Subset of respondents belonging to top clusters.
        model (SentenceTransformer): Pretrained embedding model.

    Returns:
        dict[str, np.ndarray]: Mapping from cluster_code to average embedding vector.
    """

    logging.info("Computing base cluster embeddings...")
    cluster_embeds = {}
    for cluster_code, group in df.groupby("cluster_code"):
        texts = group["clean_text"].tolist()
        embeds = model.encode(texts)
        cluster_embeds[cluster_code] = np.mean(embeds, axis=0)
    return cluster_embeds

def score_similarity(base_cluster_vecs, target_vec, row, alpha=0.7, beta=0.3):
    """
    Score similarity between a target response and base clusters using a weighted blend 
    of cosine similarity and feature-level overlap (method/gender/role).

    Parameters:
        base_cluster_vecs (dict): Precomputed cluster embeddings.
        target_vec (np.ndarray): Embedding vector of the current text.
        row (pd.Series): Metadata for the current respondent.
        alpha (float): Weight for cosine similarity.
        beta (float): Weight for feature overlap score.

    Returns:
        tuple:
            - best_cluster (str): Cluster code with highest similarity.
            - best_score (float): Corresponding similarity score.
    """

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
    """
    Assign respondents to the most similar base cluster.

    Parameters:
        df (pd.DataFrame): Respondents not in top clusters.
        base_codes (list): List of cluster codes for base clusters.
        base_cluster_vecs (dict): Cluster embeddings for base clusters.
        model (SentenceTransformer): Pretrained embedding model.

    Returns:
        pd.DataFrame: Assignments with columns [id, final_cluster, similarity_score].
    """

    logging.info("Assigning clusters based on similarity...")
    assignments = []
    for _, row in df.iterrows():
        target_vec = model.encode(row['clean_text'], convert_to_tensor=True)
        best_cluster, best_score = score_similarity(base_cluster_vecs, target_vec, row)
        assignments.append((row['id'], best_cluster, best_score))
    return pd.DataFrame(assignments, columns=['id', 'final_cluster', 'similarity_score'])

def extract_keywords(engine, df_final, model):
    """
    Extract top keywords for each final cluster and save them to SQL.

    Parameters:
        engine (sqlalchemy.Engine): Database engine for saving keywords.
        df_final (pd.DataFrame): All respondents with final cluster assignments.
        model (SentenceTransformer): Transformer model used to compute semantic scores.

    Side Effects:
        - Clears existing entries in `t_keywords_by_cluster` table.
        - Inserts new keywords and scores into the table.
    """

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
    """
    Main function to perform semantic merge of non-top clusters into top-N clusters 
    based on textual and metadata similarity.

    Parameters:
        conn_str (str): SQLAlchemy connection string.

    Side Effects:
        - Saves respondent-to-cluster map into `responden_cluster_map` table.
        - Updates `t_keywords_by_cluster` with new keyword extraction results.
        - Logs progress to a file.
    """

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
