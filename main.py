import sys, os, random, subprocess
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from pipeline import run_pipeline

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# SQL Server connection string
db_path = "mssql+pyodbc://@localhost\\SQLSERV2019/nlp_dogstrust_deploy?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
engine = create_engine(db_path)

# Get absolute path to current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

def run_script(script_name):
    script_path = os.path.join(current_dir, script_name)
    print(f"[MAIN] Running {script_name}...")
    subprocess.run([sys.executable, script_path], check=True)

if __name__ == "__main__":
    print("[MAIN] Loading raw full_text data from SQL...")
    df = pd.read_sql("""
        SELECT response_id, full_text
        FROM NLP_DOGSTRUST.dbo.t_web_cleaned
        WHERE full_text IS NOT NULL AND full_text <> ''
    """, engine)

    print(f"[MAIN] {len(df)} rows loaded. Running full pipeline...")
    run_pipeline(df, db_path)  
    print("[MAIN] Running topic modeling scripts...")
#    run_script("run_topic_method_kl.py")
#    run_script("run_topic_role_kl.py")
#    run_script("run_topic_gender_kl.py")
    run_script("run_topic_aids_kl.py")

    print("[MAIN] Running CV model scripts...")
    run_script("hybrid_semantic_gender_cv_thres.py")
    run_script("hybrid_semantic_method_cv_thres.py")
    run_script("hybrid_semantic_role_cv_thres.py")
    run_script("hybrid_semantic_aid_cv_thres.py")

    print("[MAIN] Running final prediction script...")
    run_script("hybrid_semantic_combo_predict_all_new.py")

    print("[DONE] All steps executed.")
