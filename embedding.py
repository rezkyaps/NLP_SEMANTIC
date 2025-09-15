# File: embedding.py

import os
import logging
import pandas as pd
import datetime
import json
import argparse
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from sentence_transformers import SentenceTransformer

# ========== Logging Setup ==========
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ========== Model ==========
MODEL_LOCAL = "all-mpnet-base-v2"
local_model = SentenceTransformer(MODEL_LOCAL)

def get_local_embedding(input_text: str) -> list:
    try:
        return local_model.encode(input_text).tolist()
    except Exception as e:
        logger.error(f"Local embedding failed: {e}")
        return [0.0] * 768

def generate_embeddings_with_fallback(
    conn_str: str,
    input_table: str = "trainer_profiles_clean",
    output_table: str = "trainer_profiles_embedding",
    overwrite_existing: bool = False
):
    engine = create_engine(conn_str)

    if overwrite_existing:
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {output_table}"))
        logger.info(f"Dropped table {output_table} before overwrite.")

    with engine.connect() as conn:
        df_all = pd.read_sql(f"SELECT response_id, clean_text FROM {input_table}", conn)

        if overwrite_existing:
            df = df_all.copy()
            logger.info("Overwriting all existing embeddings.")
        else:
            try:
                df_existing = pd.read_sql(f"SELECT response_id, embedding_local FROM {output_table}", conn)
                df_existing = df_existing[df_existing["embedding_local"].notnull()]
                embedded_ids = set(df_existing["response_id"])
                df = df_all[~df_all["response_id"].isin(embedded_ids)].copy()
                logger.info(f"{len(df_all)} total, {len(df)} new to embed (skipping {len(df_all) - len(df)} cached)")
            except Exception as e:
                logger.warning(f"Could not check existing table: {e}")
                df = df_all.copy()
                logger.info("Embedding all rows (output table may not exist yet).")

    local_embeddings = []
    model_used = []
    created_at = []

    for idx, row in df.iterrows():
        rid = row["response_id"]
        logger.info(f"[{idx + 1}/{len(df)}] Embedding response_id: {rid}")
        input_text = row["clean_text"]

        local_embed = get_local_embedding(input_text)

        local_embeddings.append(local_embed)
        model_used.append("local")
        created_at.append(datetime.datetime.utcnow())

    df = df.reset_index(drop=True)
    df["embedding_openai"] = None  # No OpenAI used
    df["embedding_local"] = [json.dumps(e, ensure_ascii=True) if e is not None else None for e in local_embeddings]
    df["model_used"] = model_used
    df["created_at"] = created_at
    df["response_id"] = df["response_id"].astype(str)

    logger.info(f"Saving {len(df)} new embeddings to table: {output_table}")
    df[["response_id", "embedding_openai", "embedding_local", "model_used", "created_at"]].to_sql(
        output_table, engine, if_exists="append", index=False
    )
    logger.info("Embeddings saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conn", required=True, help="SQLAlchemy connection string")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite all existing embeddings")
    parser.add_argument("--input_table", default="trainer_profiles_clean")
    parser.add_argument("--output_table", default="trainer_profiles_embedding")
    args = parser.parse_args()

    generate_embeddings_with_fallback(
        conn_str=args.conn,
        input_table=args.input_table,
        output_table=args.output_table,
        overwrite_existing=args.overwrite
    )
