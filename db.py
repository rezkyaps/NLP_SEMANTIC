# db.py
import pyodbc

conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=localhost;"
    "DATABASE=NLP_DOGSTRUST;"
    "Trusted_Connection=yes;"
)
