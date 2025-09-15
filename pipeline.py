"""
pipeline.py

Main preprocessing pipeline for text data including:
- Raw data saving
- Text cleaning
- SQL persistence
- Multiselect normalization
- Data merging
- Exploratory SQL procedures
- Text embedding generation

This module assumes input is a DataFrame containing full-text data, 
typically retrieved from SQL (e.g., `t_web_cleaned`).
"""

import sys, os
import logging
from sqlalchemy import inspect, create_engine
from cleaning import clean_text, load_custom_phrases
from normalize_multiselect import normalize_multiselect_fields
from merge import merge_data
from io_handler import save_to_sql
from eda_survey import run_eda_survey_sql, run_cluster_summary_procedure
from embedding import generate_embeddings_with_fallback

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def run_pipeline(df_raw, conn_str: str = None):
    """
    Run the full preprocessing pipeline on raw text data.

    Parameters:
    ----------
    df_raw : pandas.DataFrame
        Raw input DataFrame containing `response_id` and `full_text` columns.
        This is usually extracted from SQL table `t_web_cleaned`.

    conn_str : str, optional
        SQLAlchemy connection string to SQL Server.
        If provided, intermediate and final outputs will be stored in SQL tables.

    Steps performed:
    ----------------
    1. Save raw webscraped data to SQL (trainer_pages_raw)
    2. Clean and group full_text into clean_text by response_id
    3. Save cleaned data to SQL (trainer_profiles_clean)
    4. Normalize multiselect fields (methods, aids)
    5. Merge profiles with survey responses using semantic stratified logic
    6. Run EDA stored procedures (summary, cluster)
    7. Generate sentence-transformer embeddings with fallback prompt strategy

    Returns:
    --------
    tuple : (raw_webscraped_df, cleaned_data_df)
        - raw_webscraped_df: copy of original input
        - cleaned_data_df: processed version with clean_text and page counts
    """
    logger.info("[STEP] Starting pipeline with raw DataFrame input...")
    raw_webscraped = df_raw.copy()

    if conn_str:
        logger.info("[STEP] Saving raw webscraped data to SQL...")
        save_to_sql(raw_webscraped, "trainer_pages_raw", conn_str)
        engine = create_engine(conn_str)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        if "trainer_pages_raw" in tables:
            logger.info("Confirmed: 'trainer_pages_raw' table exists in SQL Server.")
        else:
            logger.warning("Warning: 'trainer_pages_raw' table NOT found.")

    logger.info("[STEP] Grouping and cleaning page text...")
    cleaned_data = df_raw.groupby("response_id")["full_text"].apply(lambda x: " ".join(x.dropna())).reset_index()
    cleaned_data.rename(columns={"full_text": "full_text"}, inplace=True)

    page_counts = df_raw.groupby("response_id").size().reset_index(name="page_count")
    cleaned_data = cleaned_data.merge(page_counts, on="response_id", how="left")

    filtered_phrases = load_custom_phrases(conn_str) if conn_str else []
    cleaned_data["clean_text"] = cleaned_data["full_text"].apply(lambda x: clean_text(x, filtered_phrases))

    if conn_str:
        logger.info("[STEP] Saving cleaned data to SQL...")
        save_to_sql(cleaned_data, "trainer_profiles_clean", conn_str)
        tables = inspector.get_table_names()
        if "trainer_profiles_clean" in tables:
            logger.info("Confirmed: 'trainer_profiles_clean' table exists in SQL Server.")
        else:
            logger.warning("Warning: 'trainer_profiles_clean' table NOT found.")

    logger.info("[STEP] Normalizing multiselect fields...")
    normalize_multiselect_fields(conn_str)

    logger.info("[STEP] Merging cleaned profile with survey data...")
    merge_data(conn_str)

    logger.info("[STEP] Running SQL EDA procedures...")
    run_eda_survey_sql(conn_str)
    run_cluster_summary_procedure(conn_str)

    logger.info("[STEP] Generating text embeddings...")
    generate_embeddings_with_fallback(
        conn_str,
        input_table="trainer_profiles_clean",
        output_table="trainer_profiles_embedding",
        overwrite_existing=False
    )

    logger.info("[COMPLETE] All pipeline preprocessing steps done.")
    return raw_webscraped, cleaned_data
