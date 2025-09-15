# NLP Trainer Profiling Pipeline

This project implements a full NLP-based profiling system for dog trainers using a combination of structured survey data and unstructured web text. It includes data cleaning, keyword extraction, semantic scoring, and multi-task classification.


## System Requirements

### 1. Install Anaconda
- Download: https://www.anaconda.com/products/distribution
- Then create and activate environment:

```bash
conda env create -f environment.yml
conda activate nlp_semantic
```

### 2. Install SQL Server 2019 + SSMS
- SQL Server: https://www.microsoft.com/en-us/sql-server/sql-server-downloads
- SSMS: https://learn.microsoft.com/en-us/sql/ssms/download-ssms

### 3. Run SQL Schema
- Extract `generate_sql.rar`
- Open `generate_sql.sql` in SSMS
- Execute to generate all master and transaction tables

### 3. Run SQL Schema

1. Extract `generate_sql.rar`  
2. Open `generate_sql.sql` in SSMS  
3. Execute the script to automatically generate:
   - Master tables (`m_`)
   - Transaction tables (`t_`)
   - Stored procedures (`fn_`, `sp_`)
   - Views and logic used for scoring

Why This SQL Schema Is Important???
---------------------------------

This schema defines the entire data structure used throughout the NLP profiling system. Here's why it matters:

Master Tables (`m_*`)
- Contain fixed reference data used for:
  - Mapping and interpreting labels (e.g. methods, roles, gender)
  - Controlling vocabulary for filtering (e.g. `m_filtered_words`)
  - Enforcing logical relationships (e.g. `m_method_aid_compatible`)
- Example: `m_methods_used` defines all possible dog training methods allowed in predictions and UI.

Transaction Tables (`t_*`)
- Capture raw and processed user responses, including:
  - `t_survey`: The main survey inputs
  - `t_web_cleaned`: Cleaned unstructured text (website, bio, etc.)
  - `t_keywords_by_*`: Keyword extraction & scoring results per task
  - `t_survey_methods`, `t_survey_aid`: Normalized multi-selects for easier modeling
- These are critical for reproducibility, marking, and auditing all model results.
- Example: `t_keywords_by_method` contains the scored keywords per cluster used in feature construction.

Stored Procedures (`fn_*`, `sp_*`)
- Encapsulate important business logic:
  - `fn_get_baseline_combination`: Returns the combination of methods, roles, and gender for each respondent
  - Others may compute scores, filter data, or run batch operations
- These make the pipeline modular and extensible, allowing SQL users or downstream systems (like Power BI) to query structured prediction results directly.

For Research and Evaluation
---------------------------

Running this SQL schema ensures you:
- Have consistent structure across all development, testing, and production environments
- Enable marking/labeling via normalized tables (used in cross-validation)
- Can audit model results - all predictions and keyword scores are logged to `t_*` tables
- Get end-to-end traceability from raw input to predicted label to score justification

Optional:
- Run `promptgating.sql` to set up prompt-gating scoring logic

### 4. Update DB Connection in Code
After SQL Server is installed and schema executed:
- Open `db.py`, `main.py`, and other SQL-related files
- Modify the connection string to match your local server config:

```python
connection_string = "mssql+pyodbc://<USERNAME>:<PASSWORD>@<SERVER>/<DATABASE>?driver=ODBC+Driver+17+for+SQL+Server"
```


## Project Structure & Python File Functions

### Main Pipeline
- `main.py`: Central controller script - runs end-to-end: cleaning → keyword → scoring → prediction.
- `pipeline.py`: Encapsulates the pipeline logic modularly.

### SQL Integration
- `generate_sql.rar`: Contains `generate_sql.sql`, the main schema for database setup.
- `promptgating.sql`: Adds semantic gating logic to the database.
- `test_sql_conn.py`: Test SQL connection and credential setup.
- `db.py`: Provides all SQL query and engine handling using `sqlalchemy` and `pyodbc`.

### Cleaning & Preprocessing
- `cleaning.py`: Cleans raw website content (HTML tags, symbols, filtered phrases).
- `normalize_multiselect.py`: Reshapes multi-select fields (methods, aids) into normalized format.
- `merge.py`: Merges raw text per trainer.
- `io_handler.py`: File-level saving/loading utilities.

### Survey and EDA
- `eda_survey.py`: Basic data exploration and survey distribution plots.

### Keyword Extraction & Embedding
- `embedding.py`: Loads sentence-transformers embedding model.
- `run_topic_aids_kl.py`, `run_topic_gender_kl.py`, etc: Runs keyword extraction + scoring (semantic, prompt, jsd) for each task.

### Modelling & Classification
- `hybrid_semantic_method_cv_thres.py`, etc: Cross-validation evaluation for each task (random forest classifier).
- `hybrid_semantic_combo_classifier_predict_all.py`: All-in-one multitask training pipeline.
- `hybrid_semantic_combo_predict_all_new.py`: Run predictions on new trainer data.


## How to Run

### Option 1: Full Pipeline (Recommended)
```
python main.py
```
This will:
- Clean text
- Generate and score keywords
- Extract features
- Train and predict using multi-task classifiers

### Option 2: Manual Step-by-Step Execution
1. Preprocess web and survey data:
   - `cleaning.py`, `normalize_multiselect.py`, `merge.py`
2. Generate semantic keywords:
   - `run_topic_aids_kl.py`, etc
3. Run scoring and gating (optional SQL + promptgating script)
4. Evaluate:
   - `hybrid_semantic_*_cv_thres.py`
5. Predict new profiles:
   - `hybrid_semantic_combo_predict_all_new.py`


## SQL Database Tables

### Master Tables (`m_`)
- `m_role`: Reference roles
- `m_gender`: Gender categories
- `m_methods_used`: Method list
- `m_aid_used`: Aids/tools
- `m_method_aid_compatible`: Logical mappings between aids and methods
- `m_filtered_words`: Boilerplate text to remove

### Transaction Tables (`t_`)
- `t_survey`: Survey core table
- `t_survey_methods`, `t_survey_aid`: Normalized multi-selects
- `t_web_cleaned`: Cleaned, merged web text
- `t_keywords_by_*`: Scored keywords by method, gender, aid, or role


## Output
- Cleaned text (per trainer)
- Extracted task-specific keywords
- Semantic + gated scores
- Prediction labels per task
- Final multi-task prediction output


## Notes
- Models are trained per task (method, gender, role, aid type) using random forest.
- Prompt scoring is optional but improves interpretability.
- All features (cosine, prompt, jsd) are aggregated and weighted.


## Reminder
- Ensure SQL Server is running.
- Ensure DB connection is correctly set in the Python scripts.
- You may customize output or keyword scoring weights in `pipeline.py` or `embedding.py`.

##  Notes
- Classifiers use Random Forest with weighted loss per label.
- Uses Sentence-BERT (`all-mpnet-base-v2`) for text embedding.
- Semantic, prompt, and JSD scores are aggregated as hybrid features.
- Optional scoring logic available via SQL (`promptgating.sql`).

---

##  Debugging Tips
-  Check your SQL Server is running
-  Update all `connection_string` as needed
-  Make sure `sentence-transformers` downloads are cached or online
---
This pipeline is ready for full semantic profiling of survey-website linked datasets. You can adapt the modular pieces to other multi-task NLP prediction use cases.
