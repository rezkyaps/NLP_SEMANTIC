import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sqlalchemy import create_engine
import pyodbc

# =============================================================================
# Pseudocode: Exploratory Data Analysis (EDA) for Survey Table (t_survey)
# -----------------------------------------------------------------------------
# 1. Set and validate output directory for saving plots.
# 2. Connect to SQL Server and load full survey data from 't_survey' table.
# 3. Clean column names and split multi-select fields into list (methods_used, aids_used).
# 4. Convert selected columns to categorical dtype for efficient plotting.
# 5. Explode list fields to long-form for easier aggregation.
# 6. Generate and save:
#    - Bar charts: methods, aids, roles, gender, aid types
#    - Heatmaps: gender vs method, role vs method
#    - Grouped and stacked bar charts: methods vs aids/gender/role
# 7. Final console output confirms EDA completion and output path.
# =============================================================================

def run_eda_survey_sql(conn_str, output_dir=None):
    """
    Executes exploratory data analysis (EDA) on survey data from the 't_survey' SQL table,
    generates multiple visualizations, and saves them to disk.

    Parameters:
    ----------
    conn_str : str
        SQLAlchemy-compatible connection string to the SQL Server database.

    output_dir : str or None, optional
        Directory path where the generated plots will be saved.
        If None, defaults to a folder named 'survey_eda_outputs' in the current directory.

    Returns:
    -------
    None
        The function saves multiple visualizations to the output directory, including:
            - Bar charts for method and aid proportions
            - Summary charts for categorical columns (age, gender, role, etc.)
            - Heatmaps for method usage by gender and role
            - Grouped and stacked bar charts showing the relationship between aids and methods
            - Distributions of roles, gender, and aid types
        No values are returned. Results are saved as `.png` files.
    """

    # Resolve output directory path
    if output_dir is None:
        output_dir = os.path.abspath('survey_eda_outputs')
    else:
        output_dir = os.path.abspath(output_dir)

    if isinstance(output_dir, str):
        os.makedirs(output_dir, exist_ok=True)
    else:
        raise TypeError(f"Invalid type for output_dir: {type(output_dir)} – value: {output_dir}")

    # Create SQL connection and load table
    engine = create_engine(conn_str)
    df = pd.read_sql('SELECT * FROM t_survey', engine)

    # Clean and normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Handle list-type columns safely
    df['methods_used_list'] = df['methods_used'].fillna("").str.lower().str.split(",")
    df['aids_used_list'] = df['aids_used'].fillna("").str.lower().str.split(",")

    # Cast categorical variables
    for col in ['age', 'gender', 'role', 'role_new', 'dt_staff', 'structure']:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Explode multi-label fields
    methods_exploded = df.explode('methods_used_list')
    aids_exploded = df.explode('aids_used_list')

    # ==== 1. Method proportion (bar chart)
    plt.figure(figsize=(10, 6))
    methods_exploded['methods_used_list'].value_counts(normalize=True).plot(kind='bar')
    plt.title("Proportion of Training Methods Used")
    plt.ylabel("Proportion of Respondents")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "methods_proportion.png"))
    plt.close()

    # ==== 2. Aids usage (bar chart)
    plt.figure(figsize=(12, 6))
    aids_exploded['aids_used_list'].value_counts(normalize=True).plot(kind='bar')
    plt.title("Proportion of Aids Used by Trainers")
    plt.ylabel("Proportion of Mentions")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "aids_distribution.png"))
    plt.close()

    # ==== 3. Summary plots for 6 categorical columns
    cat_cols = ['age', 'gender', 'role', 'role_new', 'dt_staff', 'structure']
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, col in zip(axes.flatten(), cat_cols):
        df[col].value_counts(normalize=True).plot(kind='bar', ax=ax)
        ax.set_title(f"Proportion of {col.replace('_', ' ').title()}")
        ax.set_ylabel("Proportion")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_proportions.png"))
    plt.close()

    # ==== 4. Gender vs. Methods Used (heatmap)
    gender_method_ct = pd.crosstab(
        methods_exploded['gender'],
        methods_exploded['methods_used_list'],
        normalize='index'
    )
    plt.figure(figsize=(10, 6))
    sns.heatmap(gender_method_ct, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Training Methods by Gender")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gender_vs_method_heatmap.png"))
    plt.close()

    # ==== 5. Role vs. Methods Used (heatmap)
    role_method_ct = pd.crosstab(
        methods_exploded['role_new'],
        methods_exploded['methods_used_list'],
        normalize='index'
    )
    plt.figure(figsize=(10, 6))
    sns.heatmap(role_method_ct, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("Training Methods by Role")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "role_vs_method_heatmap.png"))
    plt.close()

    # ==== 6. Aids Used vs. Methods Used (heatmap)
    aids_methods = pd.merge(aids_exploded[['id', 'aids_used_list']], methods_exploded[['id', 'methods_used_list']], on='id')
    aids_method_ct = pd.crosstab(
        aids_methods['aids_used_list'],
        aids_methods['methods_used_list'],
        normalize='index'
    )
     # Grouped Bar Chart for top 10 aids
    top_aids = aids_method_ct.mean(axis=1).sort_values(ascending=False).head(10).index
    plot_df = aids_method_ct.loc[top_aids].reset_index().melt(id_vars='aids_used_list')

    # Sort aids by introduce_rewards proportion for presentation
    order = aids_method_ct.loc[top_aids].sort_values(by='introduce_rewards', ascending=False).index

    plt.figure(figsize=(16, 6))
    sns.barplot(
        data=plot_df,
        x='aids_used_list',
        y='value',
        hue='methods_used_list',
        order=order
    )
    plt.ylabel("Proportion")
    plt.xlabel("Aids Used")
    plt.title("Proportion of Methods by Top Aids Used")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Training Method")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "aids_vs_method_bar.png"))
    plt.close()

    # ==== 7. Gender vs Methods (stacked bar chart)
    gender_method = pd.crosstab(methods_exploded['gender'], methods_exploded['methods_used_list'], normalize='index')
    gender_method.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title("Stacked Bar: Methods Used by Gender")
    plt.ylabel("Proportion")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gender_vs_method_stacked.png"))
    plt.close()

    # ==== 8. Role vs Methods (stacked bar chart)
    role_method = pd.crosstab(methods_exploded['role_new'], methods_exploded['methods_used_list'], normalize='index')
    role_method.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.title("Stacked Bar: Methods Used by Role")
    plt.ylabel("Proportion")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "role_vs_method_stacked.png"))
    plt.close()
    # ==== NEW BAR CHARTS ====
    # 1. Role distribution
    plt.figure(figsize=(8, 5))
    df['role_new'].value_counts(normalize=True).plot(kind='bar')
    plt.title("Distribution of Trainer Roles")
    plt.ylabel("Proportion of Respondents")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "role_distribution.png"))
    plt.close()

    # 2. Gender distribution
    plt.figure(figsize=(6, 5))
    df['gender'].value_counts(normalize=True).plot(kind='bar')
    plt.title("Distribution of Trainer Gender")
    plt.ylabel("Proportion of Respondents")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gender_distribution.png"))
    plt.close()

    # 3. Aid Type distribution (original categories)
    if 'aid_type' in df.columns:
        plt.figure(figsize=(8, 5))
        df['aid_type'].value_counts(normalize=True).plot(kind='bar')
        plt.title("Distribution of Aid Types (Original)")
        plt.ylabel("Proportion of Respondents")
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "aid_type_distribution.png"))
        plt.close()

    # 4. Aid Type Reclassified (positive, neutral, aversive)
    if 'aid_type_code' in df.columns:
        plt.figure(figsize=(8, 5))
        df['aid_type_code'].value_counts(normalize=True).plot(kind='bar')
        plt.title("Distribution of Aid Types (Reclassified)")
        plt.ylabel("Proportion of Respondents")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "aid_type_reclassified.png"))
        plt.close()
    print(" Survey EDA completed with role, gender, and aid type distributions.")

    print("\u2705 Survey EDA completed. Plots saved to:", output_dir)

def run_cluster_summary_procedure(conn_str: str):
    with pyodbc.connect(conn_str) as conn:
        cursor = conn.cursor()
        cursor.execute("EXEC sp_update_all_cluster_summary")
        cursor.commit()
        print("Cluster summary updated via stored procedure.")

# Example usage:
# run_eda_survey_sql(conn_str)
import time
import numpy as np
from sqlalchemy import create_engine

def main():
    start_time = time.time()
    seed = 56
    np.random.seed(seed)

    # SQL connection string
    conn_str = "mssql+pyodbc://@localhost\\SQLSERV2019/NLP_DOGSTRUST?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    engine = create_engine(conn_str)

    # Run EDA plots
    run_eda_survey_sql(conn_str, output_dir="survey_eda_outputs")

    # Optionally call cluster summary procedure
    try:
        run_cluster_summary_procedure(conn_str)
    except Exception as e:
        print(f" Skipping cluster summary update (error: {e})")

    elapsed = time.time() - start_time
    print(f" EDA finished in {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()

