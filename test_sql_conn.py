from sqlalchemy import create_engine, text

# Define your SQL Server connection string
conn_str = "mssql+pyodbc://@localhost/NLP_DOGSTRUST_DEPLOY?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"

def test_sql_connection(connection_string):
    """
    Test connection to a SQL Server database using SQLAlchemy.

    Parameters
    ----------
    connection_string : str
        SQLAlchemy-compatible connection string.

    Returns
    -------
    None
    """
    try:
        engine = create_engine(connection_string)
        with engine.connect() as connection:
            result = connection.execute(text("SELECT GETDATE() AS server_time;"))
            print("Connection to SQL Server succeeded.")
            for row in result:
                print(f"Server time: {row.server_time}")
    except Exception as e:
        print("Connection to SQL Server failed.")
        print(f"Error: {e}")

if __name__ == "__main__":
    test_sql_connection(conn_str)
