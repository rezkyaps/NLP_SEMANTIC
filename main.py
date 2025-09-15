import sys, os, random, subprocess
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from pipeline import run_pipeline  # Custom function to run full preprocessing pipeline

# Set random seed for reproducibility (important for consistency across runs)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Define SQL Server connection string (using Windows authentication and ODBC driver)
db_path = "mssql+pyodbc://@localhost\\SQLSERV2019/nlp_dogstrust_deploy?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
engine = create_engine(db_path)

# Get absolute path of the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define utility function to run a Python script as a subprocess
def run_script(script_name):
    """
    Executes a Python script as a subprocess from the current directory.

    Parameters:
        script_name (str): The filename of the Python script to run, 
                           assumed to be located in the same directory as this main script.

    Raises:
        subprocess.CalledProcessError: If the script exits with a non-zero status, the error is propagated.
    """
    script_path = os.path.join(current_dir, script_name)
    print(f"[MAIN] Running {script_name}...")
    subprocess.run([sys.executable, script_path], check=True)


# Main execution block
if __name__ == "__main__":

    # Step 1: Load raw full_text data from SQL Server
    print("[MAIN] Loading raw full_text data from SQL...")
    df = pd.read_sql("""
        SELECT response_id, full_text
        FROM NLP_DOGSTRUST.dbo.t_web_cleaned
        WHERE full_text IS NOT NULL AND full_text <> ''
    """, engine)

    # Step 2: Run the preprocessing pipeline (e.g. cleaning, saving, etc.)
    print(f"[MAIN] {len(df)} rows loaded. Running full pipeline...")
    run_pipeline(df, db_path)

    # Step 3: Run topic modeling scripts to extract domain-specific keywords
    print("[MAIN] Running topic modeling scripts...")
    run_script("run_topic_method_kl.py")  # For dog training method
    run_script("run_topic_role_kl.py")      # For trainer role
    run_script("run_topic_gender_kl.py")        # For gender signals
    run_script("run_topic_aids_kl.py")              # For training aids/tools

    # Step 4: Run classification model scripts with CV + thresholding
    print("[MAIN] Running CV model scripts...")
    run_script("hybrid_semantic_gender_cv_thres.py")   # Train + evaluate gender classifier
    run_script("hybrid_semantic_method_cv_thres.py")        # Train + evaluate method classifier
    run_script("hybrid_semantic_role_cv_thres.py")               # Train + evaluate role classifier
    run_script("hybrid_semantic_aid_cv_thres.py")                   # Train + evaluate aid classifier

    # Step 5: Run final combined prediction script across all dimensions
    print("[MAIN] Running final prediction script...")
    run_script("hybrid_semantic_combo_predict_all_new.py")

    # All steps completed
    print("[DONE] All steps executed.")
