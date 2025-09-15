# src/io_handler.py
import pandas as pd
import os
import logging
from sqlalchemy import create_engine

# Set up logger for IO
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def load_parquet(filename: str, base_path=None) -> pd.DataFrame:
    """
    Load a Parquet file into a Pandas DataFrame.

    Parameters
    ----------
    filename : str
        Name of the Parquet file to load.
    base_path : str, optional
        Directory where the file is located. If None, uses current file path.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    if base_path is None:
        base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, filename)
    logger.info(f"Loading parquet file from: {file_path}")
    df = pd.read_parquet(file_path)
    logger.info(f"Loaded DataFrame with shape: {df.shape}")
    return df

def save_to_sql(df: pd.DataFrame, table_name: str, conn_str: str, mode="replace"):
    """
    Save a DataFrame to a SQL Server table.

    Parameters
    ----------
    df : pd.DataFrame
        Data to save.
    table_name : str
        Target SQL table name.
    conn_str : str
        SQLAlchemy connection string.
    mode : str
        'replace' to overwrite, 'append' to add.
    """
    logger.info(f"Saving DataFrame to SQL table '{table_name}' with mode='{mode}'...")
    engine = create_engine(conn_str)
    df.to_sql(table_name, engine, if_exists=mode, index=False)
    logger.info(f"Successfully saved table '{table_name}' to database.")
