# keyword_scoring.py

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from scipy.spatial.distance import jensenshannon

def recompute_jsd_and_update(
    conn_str: str,
    table_name: str,
    code_col: str,
    w_sem_doc: float = 0.30,
    w_jsd: float = 0.20,
    w_prompt: float = 0.50,
    prompt_floor: float = 0.45,
    prompt_gate_pow: float = 2.0,
    prompt_gate_min: float = 0.30
):
    """
    Recompute final keyword score with prompt gating and update target SQL table.
    Args:
        conn_str: SQLAlchemy connection string
        table_name: Target table name (e.g., 't_keywords_by_methods')
        code_col: Column name of group code (e.g., 'method_code', 'role_code')
    """
    engine = create_engine(conn_str)
    conn = engine.raw_connection()
    cursor = conn.cursor()

    df_keywords = pd.read_sql(f"""
        SELECT {code_col}, keyword, semantic_score, semantic_prompt_score
        FROM {table_name}
    """, engine).fillna(0.0)

    group_codes = df_keywords[code_col].unique().tolist()

    # Build text blobs for JSD
    text_blobs = {}
    for code in group_codes:
        df = pd.read_sql(f"""
            SELECT w.clean_text
            FROM (
                SELECT DISTINCT id
                FROM dbo.fn_get_baseline_combination(NULL, NULL, NULL, NULL)
                WHERE {code_col} = '{code}'
            ) b
            LEFT JOIN t_web_cleaned w ON b.id = w.response_id
            WHERE w.clean_text IS NOT NULL AND w.clean_text <> ''
        """, engine)
        text_blobs[code] = " ".join(df['clean_text'].dropna().tolist()).lower()

    # Compute JSD per keyword
    jsd_result = {}
    for kw in df_keywords['keyword'].unique():
        freq = np.array([text_blobs[c].count(kw.lower()) for c in sorted(group_codes)], dtype=float)
        prob = freq / freq.sum() if freq.sum() > 0 else np.full(len(freq), 1e-10)
        uniform = np.full_like(prob, 1 / len(prob))
        jsd = 1 - jensenshannon(prob, uniform, base=2)
        jsd_result[kw] = float(jsd)

    def _clip01(x):
        x = float(x)
        if not np.isfinite(x):
            return 0.0
        return float(max(0.0, min(1.0, x)))

    # Update table row by row
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

        cursor.execute(f"""
            UPDATE {table_name}
            SET jsd_score = ?, final_score = ?
            WHERE {code_col} = ? AND keyword = ?
        """, jsd_score, final_score, row[code_col], kw)

    conn.commit()
    cursor.close()
    conn.close()
    print(f"[SCORING] Final scores updated in {table_name}.")