import pandas as pd
from sqlalchemy import create_engine

# ============================================
# PSEUDOCODE: NORMALIZE MULTI-SELECT FIELDS
# ============================================
# 1. Connect to the database using a connection string.
# 2. Read 'methods_used' and 'aids_used' columns from 't_survey' table.
# 3. Split comma-separated strings into lists for each row.
# 4. Normalize and explode each list into flat tables:
#    - One row per method -> 't_survey_methods'
#    - One row per aid    -> 't_survey_aids'
# 5. Insert both normalized tables into the SQL database.
# ============================================

def normalize_multiselect_fields(conn_str: str):
    """
    Extracts and normalizes multi-select fields ('methods_used', 'aids_used') 
    from the 't_survey' table, then inserts the normalized records into two separate 
    SQL tables: 't_survey_methods' and 't_survey_aids'.

    Parameters:
        conn_str (str): SQLAlchemy-compatible connection string to the database.

    Output:
        None. Writes two tables to the SQL database.
        - t_survey_methods (columns: id_survey, method_name)
        - t_survey_aids (columns: id_survey, aid_name)
    """
    # Establish SQL connection using SQLAlchemy
    engine = create_engine(conn_str)

    # Load raw multi-select survey fields from t_survey table
    df = pd.read_sql('SELECT id, methods_used, aids_used FROM t_survey', engine)

    # Split the comma-separated text into lists, handle nulls and lowercase normalization
    df['methods_used_list'] = df['methods_used'].fillna("").str.lower().str.split(",")
    df['aids_used_list'] = df['aids_used'].fillna("").str.lower().str.split(",")

    # Flatten the 'methods_used' list into individual rows (normalize)
    method_rows = [
        {'id_survey': row['id'], 'method_name': method.strip()}
        for _, row in df.iterrows()
        for method in row['methods_used_list'] if method.strip()
    ]
    df_methods = pd.DataFrame(method_rows)

    # Flatten the 'aids_used' list into individual rows (normalize)
    aid_rows = [
        {'id_survey': row['id'], 'aid_name': aid.strip()}
        for _, row in df.iterrows()
        for aid in row['aids_used_list'] if aid.strip()
    ]
    df_aids = pd.DataFrame(aid_rows)

    # Save normalized tables to the database (overwrite if exists)
    df_methods.to_sql("t_survey_methods", engine, if_exists="replace", index=False)
    df_aids.to_sql("t_survey_aids", engine, if_exists="replace", index=False)

    # Log confirmation
    print("t_survey_methods & t_survey_aids inserted to SQL")


# Run only if this script is executed directly
if __name__ == "__main__":
    # Define connection string to local SQL Server
    conn_str = "mssql+pyodbc://@localhost\\SQLSERV2019/NLP_DOGSTRUST?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    normalize_multiselect_fields(conn_str)
