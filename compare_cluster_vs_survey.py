import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sqlalchemy import create_engine
import json
import os

# ====================================================
# PSEUDOCODE - SURVEY & TOPIC COMPARISON ANALYSIS
# ----------------------------------------------------
# 1. Connect to SQL database and load survey responses.
# 2. Clean and normalize the column names.
# 3. Create visual summaries of:
#    - Method usage (bar chart)
#    - Gender distribution (pie chart)
#    - Age distribution (histogram)
#
# 4. Load topic prediction results (from topic modeling).
# 5. Merge with survey data on response_id.
# 6. Compare methods from survey vs predicted topics:
#    - Create a crosstab of methods vs topics.
#    - Save heatmap + CSV of this comparison.
#    - Count matches vs mismatches.
#
# 7. Plot trainer clusters (2D embeddings):
#    - First colored by predicted topic
#    - Then colored by survey-reported method
#    - Optionally draw mismatch arrows (top 100 only)
# ====================================================

def explore_survey_summary(conn_str: str, table_name: str = "t_survey"):
    """
    Loads survey data from a SQL table, performs basic cleaning and normalization, 
    and generates exploratory visualizations including method usage, gender distribution, 
    and age histogram.

    Parameters:
        conn_str (str): SQLAlchemy connection string to the database.
        table_name (str): Name of the survey table to load (default: 't_survey').

    Returns:
        pd.DataFrame: Cleaned survey dataframe with added derived fields.
    """

    engine = create_engine(conn_str)
    df = pd.read_sql_table(table_name, engine)

    # Clean columns
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Histogram for methods used
    method_counts = df["methods_used"].str.get_dummies(sep=",").sum().sort_values(ascending=False)
    method_counts.plot(kind="bar", title="Survey: Method Frequency", figsize=(10, 5))
    plt.ylabel("Number of Trainers")
    plt.tight_layout()
    plt.savefig("outputs/survey_method_barplot.png")
    plt.close()

    # Pie chart by gender per method
    if "gender" in df.columns:
        plt.figure(figsize=(6, 6))
        df["gender"].value_counts().plot.pie(autopct="%1.1f%%")
        plt.title("Gender Distribution")
        plt.ylabel("")
        plt.tight_layout()
        plt.savefig("outputs/survey_gender_pie.png")
        plt.close()

    # Age histogram
    if "age" in df.columns:
        plt.figure(figsize=(8, 5))
        df["age"].dropna().astype(int).plot.hist(bins=10)
        plt.title("Trainer Age Distribution")
        plt.xlabel("Age")
        plt.tight_layout()
        plt.savefig("outputs/survey_age_hist.png")
        plt.close()

    return df


def compare_methods_survey_vs_cluster(survey_df: pd.DataFrame, cluster_df: pd.DataFrame, topic_model_path: str):
    """
    Compares survey-reported training methods with predicted topic clusters, 
    generates a heatmap and mismatch summary.

    Parameters:
        survey_df (pd.DataFrame): DataFrame containing raw survey data.
        cluster_df (pd.DataFrame): DataFrame with topic model predictions.
        topic_model_path (str): Output directory for saving plots and CSVs.

    Returns:
        pd.DataFrame: Merged DataFrame with survey methods and topic predictions.
    """

    # Merge survey with topic prediction
    df = pd.merge(survey_df, cluster_df, on="response_id", how="inner")

    # Normalize columns
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.dropna(subset=["topic"])

    # Show topic vs method table
    df["methods_used"] = df["methods_used"].fillna("None")
    df["methods_used_list"] = df["methods_used"].str.lower().str.split(",")
    exploded = df.explode("methods_used_list")

    ct = pd.crosstab(exploded["methods_used_list"], exploded["topic"])
    ct.to_csv("outputs/survey_vs_cluster_matrix.csv")

    plt.figure(figsize=(10, 6))
    sns.heatmap(ct, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Survey Method vs Predicted Topic")
    plt.tight_layout()
    plt.savefig("outputs/survey_vs_cluster_heatmap.png")
    plt.close()

    # Logging mismatch
    mismatch_count = (exploded["methods_used_list"] != exploded["topic"].astype(str)).sum()
    match_count = len(exploded) - mismatch_count
    with open("outputs/match_mismatch_summary.txt", "w") as f:
        f.write(f"Match count: {match_count}\n")
        f.write(f"Mismatch count: {mismatch_count}\n")

    return df


def plot_cluster_colored_by_survey(df: pd.DataFrame, embedding_col: str = "embedding_local"):
    """
    Visualizes 2D cluster projections of trainers based on embeddings.
    Generates two scatter plots:
        1. Colored by predicted topic.
        2. Colored by reported survey methods.
    Also includes optional mismatch annotation arrows.

    Parameters:
        df (pd.DataFrame): DataFrame with 2D embeddings and method/topic info.
        embedding_col (str): Column containing serialized embedding vectors (default: 'embedding_local').

    Returns:
        None. Saves visualizations as PNG files to disk.
    """

    # Extract 2D embeddings for plotting
    coords = np.vstack(df[embedding_col].apply(json.loads).tolist())
    topics = df["topic"]
    coords = np.vstack(df[embedding_col].apply(json.loads).tolist())

    # FILTER: remove topic -1 (outlier) from plot
    mask = topics != -1
    coords = coords[mask]
    topics = topics[mask]
    methods = df.loc[mask, "methods_used"].fillna("None")

    methods = df["methods_used"].fillna("None")

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=topics, cmap="tab10", alpha=0.7, edgecolors="w")
    plt.title("Trainer Clusters (colored by Predicted Topic)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(*scatter.legend_elements(), title="Topic")
    plt.tight_layout()
    plt.savefig("outputs/cluster_colored_topics.png")
    plt.close()

    method_palette = sns.color_palette("tab10", n_colors=len(set(methods)))
    method_map = {m: method_palette[i] for i, m in enumerate(set(methods))}
    colors = methods.map(method_map)

    plt.figure(figsize=(10, 8))
    for i, method in enumerate(set(methods)):
        mask = methods == method
        plt.scatter(coords[mask, 0], coords[mask, 1], label=method, alpha=0.5, edgecolors="w")

    plt.title("Trainer Clusters Colored by Survey Methods")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(title="Survey Method", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("outputs/cluster_colored_methods.png")
    plt.close()

    # Optional: draw mismatch connections (top 100 for readability)
    mismatches = df[df["methods_used_list"].str.contains("None") == False].head(100)
    for i in range(len(mismatches)):
        x, y = coords[i]
        plt.annotate("", xy=(x, y), xytext=(x + 0.2, y + 0.2), arrowprops=dict(arrowstyle="->", color='gray', alpha=0.3))

    plt.tight_layout()
    plt.savefig("outputs/cluster_arrows_mismatch.png")
    plt.close()


def run_analysis(conn_str: str):
    """
    Executes the full analysis pipeline:
        - Loads topic cluster predictions and survey data
        - Compares topics vs methods
        - Generates cluster visualizations

    Parameters:
        conn_str (str): SQLAlchemy connection string to the database.

    Returns:
        None. Saves all outputs to the 'outputs/' directory.
    """

    topic_model_path = "outputs"

    engine = create_engine(conn_str)
    cluster_df = pd.read_sql("SELECT * FROM trainer_profiles_topics", engine)
    survey_df = explore_survey_summary(conn_str)
    df = compare_methods_survey_vs_cluster(survey_df, cluster_df, topic_model_path)
    plot_cluster_colored_by_survey(df)
