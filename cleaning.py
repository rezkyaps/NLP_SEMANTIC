# src/cleaning.py
import re
import logging
import pandas as pd
from sqlalchemy import create_engine

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
########################################
# PSEUDOCODE FOR CLEANING MODULE
########################################

# 1. Set up a logger to track process information or issues.
# 2. Define a function to load custom spam phrases from a SQL table.
#    - Connect to database.
#    - Load 'phrase' column from 'filtered_words' table.
#    - Return list of non-null phrases.
#
# 3. Define a function to clean a text string:
#    - Remove spam phrases if any provided.
#    - Remove redacted markers, HTML tags, and content in brackets.
#    - Remove error messages like 404.
#    - Remove non-printable characters, symbols, and URLs.
#    - Remove UK phone numbers and other known spam patterns.
#    - Normalize: lowercase, remove special characters, collapse whitespace.
#    - Return cleaned text string.

def load_custom_phrases(conn_str: str, table_name: str = "filtered_words") -> list:
    """
    Loads a list of custom phrases (e.g., spam or noise) from a SQL table for filtering.

    Parameters:
        conn_str (str): SQLAlchemy connection string to the database.
        table_name (str): Name of the SQL table containing the 'phrase' column (default: 'filtered_words').

    Returns:
        list: List of strings representing phrases to filter from text.
              Returns an empty list if loading fails.
    """

    """
    Load spam phrases from SQL table [filtered_words].
    """
    try:
        engine = create_engine(conn_str)
        with engine.connect() as conn:
            df = pd.read_sql_table(table_name, conn)
            phrases = df['phrase'].dropna().astype(str).tolist()
            logger.info(f" Loaded {len(phrases)} phrases from SQL table '{table_name}'.")
            return phrases
    except Exception as e:
        logger.error(f" Failed to load phrases from SQL: {e}")
        return []

def clean_text(text: str, custom_phrases: list = None) -> str:
    """
    Cleans a given text string by applying several filters such as:
    removing spam phrases, redacted content, HTML tags, phone numbers,
    known spam terms, and normalizing the format.

    Parameters:
        text (str): Raw input text to be cleaned.
        custom_phrases (list, optional): List of phrases to remove from the text.

    Returns:
        str: Cleaned and normalized text string.
    """

    """
    Clean scraped text by removing spam phrases, redacted markers, HTML, and noise.
    """
    if not isinstance(text, str):
        return ""

    original_length = len(text)

    # Remove spam phrases
    if custom_phrases:
        for phrase in custom_phrases:
            pattern = re.escape(phrase.strip().lower())
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Remove content in square or curly brackets
    text = re.sub(r"\[[^\]]*redacted[^\]]*\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\{[^}]*\}", "", text)  # remove {...}

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove 404 and similar errors
    text = re.sub(r"error\s+404.*?(?:\.|\n|$)", "", text, flags=re.IGNORECASE)

    # Remove non-printable characters
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)

    # Lowercase and normalize
    text = text.lower()

    # Remove double or single pipes
    text = text.replace("||", " ")
    text = text.replace("|", " ")

    # Remove trademark symbols
    text = re.sub(r"[®©™]", "", text)

    # Remove URLs and domains
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove UK phone numbers
    text = re.sub(r"\b(?:\+44\s?|\(?0\d{3,4}\)?[\s-]?)\d{3,4}[\s-]?\d{3,4}\b", "", text)

    # Remove common spam keywords
    text = re.sub(r"\b(contact|services|email|phone|copyright|postcode|message|get in touch|t&c's)\b", "", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    cleaned_length = len(text)
    if cleaned_length < original_length:
        logger.debug(f"Text cleaned. Length: {original_length} → {cleaned_length}")

    return text
