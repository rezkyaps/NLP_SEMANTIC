import pandas as pd
from sqlalchemy import create_engine

def normalize_multiselect_fields(conn_str: str):
    engine = create_engine(conn_str)

    df = pd.read_sql('SELECT id, methods_used, aids_used FROM t_survey', engine)
    df['methods_used_list'] = df['methods_used'].fillna("").str.lower().str.split(",")
    df['aids_used_list'] = df['aids_used'].fillna("").str.lower().str.split(",")

    method_rows = [
        {'id_survey': row['id'], 'method_name': method.strip()}
        for _, row in df.iterrows()
        for method in row['methods_used_list'] if method.strip()
    ]
    df_methods = pd.DataFrame(method_rows)

    aid_rows = [
        {'id_survey': row['id'], 'aid_name': aid.strip()}
        for _, row in df.iterrows()
        for aid in row['aids_used_list'] if aid.strip()
    ]
    df_aids = pd.DataFrame(aid_rows)

    df_methods.to_sql("t_survey_methods", engine, if_exists="replace", index=False)
    df_aids.to_sql("t_survey_aids", engine, if_exists="replace", index=False)

    print("t_survey_methods & t_survey_aids inserted to SQL")


if __name__ == "__main__":
    conn_str = "mssql+pyodbc://@localhost\\SQLSERV2019/NLP_DOGSTRUST?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    normalize_multiselect_fields(conn_str )
