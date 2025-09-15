
"""
End-to-end multitask predictor with robust gender mapping (03 -> 02) and
safe keyword-feature fallback to avoid concatenate errors when keywords are missing.

Why these patches:
- Merge lbl_03_gender into lbl_02_gender globally (training + evaluation stay 2-label).
- Drop gender code '03' keywords (not needed) and handle empty keyword table per task.
- Guard np.concatenate when no arrays are available (fallback to zero-feature column).
"""

from __future__ import annotations

# --- Imports
from itertools import combinations
from collections import Counter
import logging
import warnings
import os
import json

import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    roc_curve,
    auc as compute_auc,
    f1_score,
    accuracy_score,
    hamming_loss,
    jaccard_score,
    roc_auc_score,
)

from sqlalchemy import create_engine, text
from sqlalchemy import Table, Column, Integer, String, MetaData, Boolean, Float
##############################################
# PSEUDOCODE - HYBRID SEMANTIC COMBO PREDICT
##############################################

# 1. Load all survey responses with labels from SQL.
# 2. Merge gender label 03 into 02 (binary only).
# 3. Encode survey texts into sentence embeddings using SBERT.
# 4. For each prediction task (method, gender, role, aid_type):
#     a. Load keywords from SQL and compute semantic similarity.
#     b. Generate padded, weighted feature vectors.
#     c. Run CV to tune class weights and thresholds (per label).
#     d. Train one RandomForest per label (or load from disk).
#     e. Predict probabilities and apply thresholds to get predictions.
#     f. (If single-label task) apply argmax masking strategy.
#     g. Save per-task metrics and store predictions.
# 5. Compute per-respondent correctness across all tasks.
# 6. Log and save:
#     - Full predictions table
#     - Per-task evaluation scores
#     - Exhaustive combination evaluation
#     - Greedy stepwise task addition (to measure incremental gain)
# 7. Persist stepwise task match matrix and plot interactive metrics chart.

# --- Logging
preds_dict: dict[str, np.ndarray] = {}
y_true_dict: dict[str, np.ndarray] = {}

logging.basicConfig(
    filename="combo_classifier_predict.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def log(step: str) -> None:
    print(f"\033[92m[INFO]\033[0m {step}...")
    logging.info(step)


# --- Config
LABEL_GROUPS: dict[str, list[str]] = {
    "method": ["lbl_01_survey", "lbl_02_survey", "lbl_03_survey", "lbl_04_survey"],
    "gender": ["lbl_01_gender", "lbl_02_gender"],  # 2 labels only (03 merged into 02)
    "role": ["lbl_01_role", "lbl_02_role", "lbl_03_role"],
    "aid_type": ["lbl_01_aid_type", "lbl_02_aid_type", "lbl_03_aid_type"],
}

TABLE_MAP = {
    "method": "t_key_report_method",
    "role": "t_key_report_role",
    "gender": "t_key_report_gender",
    "aid_type": "t_key_report_aidtype",
}

REG_PARAMS = {
    "method": {
        "max_depth": 10,             # butuh tree dalam untuk 1000+ fitur
        "min_samples_split": 8,      # supaya bisa split lebih sering
        "min_samples_leaf": 3,       # tangkap edge cases
        "n_estimators": 30,          # lebih banyak pohon
        "max_features": "sqrt",
        "ccp_alpha": 0.01,           # pruning ringan biar gak overfit
    },
    "role": {
        "max_depth": 10,             # fitur sedang (500an)
        "min_samples_split": 8,
        "min_samples_leaf": 3,
        "n_estimators": 30,
        "max_features": "sqrt",
        "ccp_alpha": 0.005,
    },
    "gender": {
        "max_depth": 6,              # simple binary, fitur sedikit
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "n_estimators": 25,
        "max_features": "sqrt",
        "ccp_alpha": 0.001,
    },
    "aid_type": {
        "max_depth": 6,
        "min_samples_split": 8,
        "min_samples_leaf": 3,
        "n_estimators": 25,
        "max_features": "sqrt",
        "ccp_alpha": 0.005,
    },
}

CODE_COLUMN_MAP = {
    "method": "method_code",
    "role": "role_code",
    "gender": "gender_code",
    "aid_type": "aid_type_code",
}


# --- Helpers

def remap_gender_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merges lbl_03_gender into lbl_02_gender using logical OR, and zeros out lbl_03_gender.

    This transformation standardizes the label space to 2 gender classes (01 and 02),
    as required by downstream modeling and evaluation logic.

    Parameters:
        df (pd.DataFrame): Input DataFrame with gender label columns.

    Returns:
        pd.DataFrame: Modified DataFrame with merged gender labels.
    """

    
    """Merge lbl_03_gender into lbl_02_gender (03 -> 02), keep 2 columns only.
    Why: Requirement says 03 is merged to 02; keywords for 03 are not needed.
    """
    if {"lbl_01_gender", "lbl_02_gender", "lbl_03_gender"}.issubset(df.columns):
        # Merge as logical OR; cast to int to keep 0/1
        df["lbl_02_gender"] = (df["lbl_02_gender"].astype(int) | df["lbl_03_gender"].astype(int)).astype(int)
        df["lbl_03_gender"] = 0  # Keep column to avoid SQL surprises elsewhere; value 0
    return df


def safe_keyword_feature_matrix(
    model_st: SentenceTransformer,
    text_vecs: np.ndarray,
    df_kw: pd.DataFrame,
    code_col: str,
    task: str,
) -> np.ndarray:
    """
    Generates semantic similarity-based feature matrix between input texts and task-specific keywords.

    Applies scoring weights, zero-padding, and handles:
    - Missing keywords
    - Redundant gender code '03'

    Parameters:
        model_st (SentenceTransformer): Preloaded embedding model.
        text_vecs (np.ndarray): Sentence embeddings of input texts.
        df_kw (pd.DataFrame): Keyword table with semantic scores.
        code_col (str): Column containing the class codes (e.g., 'aid_type_code').
        task (str): Name of the classification task.

    Returns:
        np.ndarray: Semantic feature matrix (n_samples x n_features).
    """

    n_samples = len(text_vecs)

    for col in ['semantic_score', 'final_score', 'gate_factor', 'semantic_prompt_score']:
        if col in df_kw.columns:
            df_kw[col] = df_kw[col].astype(str).str.replace(',', '.').astype(float)

    if task == "gender":
        df_kw = df_kw[df_kw[code_col] != "03"]

    if df_kw.empty:
        log(f"[PATCH] No keywords for task={task}. Using zero features.")
        return np.zeros((n_samples, 1), dtype=np.float32)

    label_codes = sorted(df_kw[code_col].unique().tolist())
    kw_by_code = df_kw.groupby(code_col)

    keyword_vecs = {code: model_st.encode(kw_by_code.get_group(code)["keyword"].tolist(), show_progress_bar=False)
                    for code in label_codes}

    score_map = {col: {code: kw_by_code.get_group(code)[col].values for code in label_codes}
                 for col in ['semantic_score', 'final_score', 'gate_factor', 'semantic_prompt_score'] if col in df_kw.columns}

    max_kw = int(df_kw.groupby(code_col).size().max())

    features: list[np.ndarray] = []
    for vec in text_vecs:
        row_parts: list[np.ndarray] = []
        for code in label_codes:
            sims = cosine_similarity([vec], keyword_vecs[code])[0]

            padded_blocks = []
            for col in score_map:
                weighted = sims * score_map[col][code]
                padded = np.pad(weighted, (0, max_kw - len(weighted)), mode="constant")[:max_kw]
                padded_blocks.append(padded)

            mean_sim = np.mean(sims)
            max_sim = np.max(sims)
            presence_flag = float(np.any(sims > 0.5))

            row_parts.append(np.concatenate(padded_blocks + [[mean_sim, max_sim, presence_flag]]))

        features.append(np.concatenate(row_parts) if row_parts else np.zeros((1,), dtype=np.float32))

    return np.asarray(features, dtype=np.float32)


def save_detailed_predictions(final_results: pd.DataFrame,
                              y_true_dict: dict[str, np.ndarray],
                              preds_dict: dict[str, np.ndarray],
                              tasks: list[str],
                              engine) -> None:
    """
    Saves respondent-level predictions and ground truth per task to SQL table,
    including per-label mismatches and overall match flags.

    Parameters:
        final_results (pd.DataFrame): DataFrame containing response_id and predictions.
        y_true_dict (dict): Ground truth label arrays per task.
        preds_dict (dict): Predicted label arrays per task.
        tasks (list): List of task names.
        engine (SQLAlchemy engine): Active SQLAlchemy connection engine.
    """

    metadata = MetaData()
    table_name = "t_detailed_preds"

    # Create table if not exists
    columns = [
        Column("response_id", String, primary_key=True),
        Column("hard_match", Boolean),
        Column("mismatch_count", Integer),
        Column("mismatch_tasks", String),
    ]
    for task in tasks:
        for j, label in enumerate(LABEL_GROUPS[task]):
            columns.append(Column(f"true_{label}", Integer))
            columns.append(Column(f"pred_{label}", Integer))

    table = Table(table_name, metadata, *columns)
    metadata.create_all(engine)

    log("Saving detailed predictions with ground truth + prediction per task")

    mismatch_rows: list[dict] = []
    for i, rid in enumerate(final_results["response_id"]):
        row = {"response_id": rid}
        mismatch_tasks: list[str] = []
        mismatch_count = 0
        hard_match = True

        for task in tasks:
            y_true = y_true_dict[task][i]
            y_pred = preds_dict[task][i]
            is_match = np.array_equal(y_true, y_pred)
            if not is_match:
                mismatch_tasks.append(task)
                mismatch_count += 1
                hard_match = False

            for j, label in enumerate(LABEL_GROUPS[task]):
                row[f"true_{label}"] = int(y_true[j])
                row[f"pred_{label}"] = int(y_pred[j])

        row["hard_match"] = hard_match
        row["mismatch_count"] = mismatch_count
        row["mismatch_tasks"] = ", ".join(mismatch_tasks)
        mismatch_rows.append(row)

        # Console trace per respondent (why: easy audit)
        log(f"RESP: {rid} | HARD_MATCH={hard_match} | MISMATCH={mismatch_tasks if mismatch_tasks else '-'}")
        for task in tasks:
            y_true = y_true_dict[task][i]
            y_pred = preds_dict[task][i]
            label_names = LABEL_GROUPS[task]
            for j, label in enumerate(label_names):
                log(f"\t{task.upper()} | {label}: TRUE={y_true[j]} | PRED={y_pred[j]}")

    df_mismatch = pd.DataFrame(mismatch_rows)
    df_mismatch.to_sql(table_name, engine, schema="dbo", if_exists="replace", index=False)
    log(f"Saved detailed respondent-level predictions to DB table: {table_name}")


def save_single_task_scores(task: str, y_true: np.ndarray, y_pred: np.ndarray, engine) -> None:
    """
    Computes and saves macro F1 and accuracy for a single task to SQL.

    Parameters:
        task (str): Name of the prediction task.
        y_true (np.ndarray): Ground truth binary matrix.
        y_pred (np.ndarray): Predicted binary matrix.
        engine (SQLAlchemy engine): SQL connection engine.
    """

    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    df_score = pd.DataFrame([
        {
            'task': task,
            'macro_f1': f1,
            'accuracy': acc,
            'n_samples': len(y_true),
            'created_at': pd.Timestamp.now(),
        }
    ])

    df_score.to_sql("t_single_task_scores", engine, schema="dbo", if_exists="append", index=False)
    log(f"Saved single-task score for '{task}' to DB: F1={f1:.4f}, ACC={acc:.4f}")


# Visualization of stepwise metrics
def plot_stepwise_metrics_interactive(engine, output_path="E:/stepwise_metrics.html") -> None:
    """
    Generates and saves an interactive Plotly chart of stepwise macro F1 and accuracy gains.

    Parameters:
        engine (SQLAlchemy engine): SQL connection engine.
        output_path (str): Path to save the HTML chart.
    """

    import os
    import pandas as pd
    import plotly.graph_objects as go

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_metrics = pd.read_sql("SELECT * FROM t_stepwise_metrics", engine)

    fig = go.Figure()

    # Accuracy
    fig.add_trace(go.Scatter(
        x=df_metrics["step"],
        y=df_metrics["accuracy"],
        mode="lines+markers",
        name="Accuracy",
        marker=dict(size=10),
        line=dict(width=3),
    ))

    # Macro F1
    fig.add_trace(go.Scatter(
        x=df_metrics["step"],
        y=df_metrics["macro_f1"],
        mode="lines+markers",
        name="Macro F1",
        marker=dict(size=10),
        line=dict(width=3, dash='dash'),
    ))

    # Gain bars
    fig.add_trace(go.Bar(x=df_metrics["step"], y=df_metrics["gain_acc"], name="Gain Accuracy", opacity=0.6))
    fig.add_trace(go.Bar(x=df_metrics["step"], y=df_metrics["gain_f1"], name="Gain F1", opacity=0.6))

    # Annotations for task labels
    annotations = []
    for i, row in df_metrics.iterrows():
        annotations.append(dict(
            x=row["step"],
            y=max(row["accuracy"], row["macro_f1"]) + 0.035,
            text=row["task"],
            showarrow=False,
            font=dict(size=18, color="black"),
            xanchor="center",
            yanchor="bottom",
        ))

    # Clean step labels (remove comma if any)
    df_metrics["step_clean"] = df_metrics["step"].str.replace(",", "", regex=False)

    fig.update_layout(
        title="Stepwise Combo Metrics (Interactive)",
        xaxis=dict(
            title="Step",
            tickmode='array',
            tickvals=df_metrics["step"],
            ticktext=df_metrics["step_clean"],
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            title="Score",
            tickfont=dict(size=16),
            titlefont=dict(size=18)
        ),
        legend=dict(font=dict(size=14)),
        barmode="group",
        template="plotly_white",
        font=dict(size=14),
        annotations=annotations
    )

    fig.write_html(output_path)
    print(f"Interactive stepwise metrics plot saved to {output_path}")






# CV + thresholds per label

def run_cv_tuning_and_save(task: str, X: np.ndarray, y_true: np.ndarray, response_ids: list[str], engine) -> tuple:
    """
    Runs single-pass CV to tune thresholds and class weights per label.

    Also logs and saves results into SQL (t_model_thresholds), versioned per task.

    Parameters:
        task (str): Task name (e.g., 'method', 'gender').
        X (np.ndarray): Feature matrix.
        y_true (np.ndarray): Ground truth label matrix.
        response_ids (list): List of respondent IDs.
        engine (SQLAlchemy engine): SQL engine.

    Returns:
        tuple: thresholds, class_weights, AUCs, threshold_map (dict[label_index] -> float)
    """

    log(f"[CV TUNING] Running CV for task: {task}")

    # Legacy safety: if gender still has 3 cols, merge 03 -> 02 here as well
    if task == "gender" and y_true.shape[1] == 3:
        log("[PATCH] CV: Merging lbl_03_gender into lbl_02_gender")
        for i in range(len(y_true)):
            if y_true[i, 2] == 1:
                y_true[i, 2] = 0
                y_true[i, 1] = 1
        y_true = y_true[:, :2]

    thresholds: list[float] = []
    class_weights: list[float] = []
    aucs: list[float] = []

    for i in range(y_true.shape[1]):
        y = y_true[:, i]
        if len(np.unique(y)) < 2:
            thresholds.append(0.5)
            class_weights.append(1.0)
            aucs.append(0.0)
            log(f"[CV-{task}] Label {i} SKIPPED — Only one class present")
            continue

        weight_arr = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
        weight = float(weight_arr[1]) if len(weight_arr) > 1 else 1.0
        class_weights.append(weight)

        rf = RandomForestClassifier(
            class_weight={0: 1.0, 1: weight},
            max_depth=REG_PARAMS[task]["max_depth"],
            min_samples_split=REG_PARAMS[task]["min_samples_split"],
            min_samples_leaf=REG_PARAMS[task]["min_samples_leaf"],
            max_features='sqrt',
            n_estimators=REG_PARAMS[task]["n_estimators"],
            random_state=56,
            #ccp_alpha=0.01
        )
        rf.fit(X, y)

        try:
            probas = rf.predict_proba(X)
            if probas.shape[1] == 1:
                fold_probs = np.full(X.shape[0], 1.0 if rf.classes_[0] == 1 else 0.0)
            else:
                fold_probs = probas[:, 1]
        except Exception:
            fold_probs = np.zeros(X.shape[0])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            fpr, tpr, thres = roc_curve(y, fold_probs)
            youden = tpr - fpr
            best_thresh = thres[np.argmax(youden)] if len(thres) > 0 else 0.5
            #best_thresh=0.5

        auc_val = compute_auc(fpr, tpr) if len(fpr) > 0 else 0.0
        thresholds.append(float(best_thresh))
        aucs.append(float(auc_val))

        log(f"[CV-{task}] Label {i} | AUC: {auc_val:.3f} | Threshold: {best_thresh:.3f} | Weight: {weight:.3f}")

        # Persist threshold & weight (versioned)
        with engine.begin() as conn:
            max_version = conn.execute(
                text("SELECT ISNULL(MAX(model_version), '0') FROM NLP_DOGSTRUST.dbo.t_model_thresholds WHERE task_name = :task_name"),
                {"task_name": task},
            ).scalar()
            new_version = str(int(max_version) + 1)

            label_name = LABEL_GROUPS[task][i]
            conn.execute(
                text(
                    """
                    INSERT INTO NLP_DOGSTRUST.dbo.t_model_thresholds
                    (task_name, label_index, label_name, threshold, class_weight, model_version, created_at)
                    VALUES (:task_name, :label_index, :label_name, :threshold, :class_weight, :model_version, GETDATE())
                    """
                ),
                {
                    "task_name": task,
                    "label_index": str(i),
                    "label_name": label_name,
                    "threshold": str(best_thresh),
                    "class_weight": str(weight),
                    "model_version": new_version,
                },
            )

    log(f"[CV-{task}] Saved {len(thresholds)} rows to DB under version={new_version}")
    threshold_map = {i: thresholds[i] for i in range(len(thresholds))} 
    return thresholds, class_weights, aucs, threshold_map



# Train & predict utilities
def apply_threshold(probas: np.ndarray, threshold_map: dict[int, float], default_threshold: float = 0.5) -> np.ndarray:
    """
    Applies per-label thresholds to probability matrix to generate binary predictions.

    Parameters:
        probas (np.ndarray): Predicted probabilities (n_samples x n_labels).
        threshold_map (dict): Map of thresholds per label index.
        default_threshold (float): Fallback threshold if not found in map.

    Returns:
        np.ndarray: Binary prediction matrix.
    """

    preds = np.zeros_like(probas, dtype=int)
    for i in range(probas.shape[1]):
        threshold = threshold_map.get(i, default_threshold)
        preds[:, i] = (probas[:, i] >= threshold).astype(int)
    return preds

def train_per_label_rf(task: str, X: np.ndarray, y_true: np.ndarray, weights: list[dict[int, float]], seed: int, model_path: str):
    """
    Trains one RandomForestClassifier per label in the task and saves models to disk.

    Parameters:
        task (str): Task name.
        X (np.ndarray): Feature matrix.
        y_true (np.ndarray): Ground truth matrix.
        weights (list): List of class_weight dicts per label.
        seed (int): Random seed.
        model_path (str): File path to save the trained model list.

    Returns:
        list: List of trained classifier models.
    """

    models = []
    for i in range(y_true.shape[1]):
        clf = RandomForestClassifier(random_state=seed, max_depth=REG_PARAMS[task]["max_depth"],
            min_samples_split=REG_PARAMS[task]["min_samples_split"],
            min_samples_leaf=REG_PARAMS[task]["min_samples_leaf"],
            max_features='sqrt',
            n_estimators=REG_PARAMS[task]["n_estimators"],
            
            )
        if weights and i < len(weights):
            try:
                clf.set_params(class_weight=weights[i])
            except Exception:
                pass
        clf.fit(X, y_true[:, i])
        models.append(clf)
    joblib.dump(models, model_path)
    log(f"[INFO] Per-label RandomForest models trained and saved to {model_path}")
    return models


def predict_proba_per_label(models, X: np.ndarray) -> np.ndarray:
    """
    Runs prediction using a list of per-label classifiers and returns probabilities.

    Parameters:
        models (list): List of trained models (one per label).
        X (np.ndarray): Input feature matrix.

    Returns:
        np.ndarray: Predicted probabilities (n_samples x n_labels).
    """

    probas = []
    for clf in models:
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(X)
            classes = getattr(clf, "classes_", np.array([0, 1]))
            if proba.shape[1] == 1:
                col = np.full((X.shape[0],), 1.0) if classes[0] == 1 else np.full((X.shape[0],), 0.0)
            else:
                if 1 in classes:
                    idx = int(np.where(classes == 1)[0][0])
                    col = proba[:, idx]
                else:
                    col = np.zeros(X.shape[0])
            probas.append(col)
        else:
            probas.append(np.zeros(X.shape[0]))
    return np.stack(probas, axis=1)


def safe_roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Computes ROC AUC, returns 0.5 if only one class is present.

    Parameters:
        y_true (np.ndarray): Binary ground truth.
        y_score (np.ndarray): Predicted probabilities.

    Returns:
        float: AUC score or fallback.
    """

    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def evaluate_combo(preds_dict: dict[str, np.ndarray], y_true_dict: dict[str, np.ndarray], tasks: list[str]) -> dict[str, float]:
    """
    Evaluates multi-task predictions by combining specified tasks and computing metrics.

    Parameters:
        preds_dict (dict): Predicted labels per task.
        y_true_dict (dict): Ground truth labels per task.
        tasks (list): Subset of task names to evaluate jointly.

    Returns:
        dict: Dictionary with match accuracy, macro F1, and macro accuracy.
    """

    y_true_concat = np.concatenate([y_true_dict[task] for task in tasks], axis=1)
    y_pred_concat = np.concatenate([preds_dict[task] for task in tasks], axis=1)

    hard_match = np.all(y_true_concat == y_pred_concat, axis=1)
    match_acc = float(np.mean(hard_match))

    macro_f1 = float(f1_score(y_true_concat, y_pred_concat, average="macro", zero_division=0))
    macro_acc = float(accuracy_score(y_true_concat, y_pred_concat))

    return {"match_acc": match_acc, "macro_f1": macro_f1, "macro_acc": macro_acc}

def oof_predict_per_label_rf(task: str, X: np.ndarray, y_true: np.ndarray, weights: list[dict[int, float]], seed: int, n_splits: int = 5):
    """
    Performs out-of-fold predictions using stratified K-Fold CV per label.

    Parameters:
        task (str): Task name.
        X (np.ndarray): Feature matrix.
        y_true (np.ndarray): Ground truth label matrix.
        weights (list): List of class weights per label.
        seed (int): Random seed for reproducibility.
        n_splits (int): Number of folds.

    Returns:
        tuple: (OOF predictions matrix, list of models per label)
    """

    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone

    n_samples, n_labels = y_true.shape
    oof_preds = np.zeros((n_samples, n_labels))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    models_per_label = [[] for _ in range(n_labels)]

    for i in range(n_labels):
        y_label = y_true[:, i]
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_label)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_label[train_idx], y_label[val_idx]

            clf = RandomForestClassifier(
                class_weight=weights[i],
                max_depth=REG_PARAMS[task]["max_depth"],
                min_samples_split=REG_PARAMS[task]["min_samples_split"],
                min_samples_leaf=REG_PARAMS[task]["min_samples_leaf"],
                max_features='sqrt',
                n_estimators=REG_PARAMS[task]["n_estimators"],
                random_state=seed + fold,
            )
            clf.fit(X_train, y_train)

            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X_val)
                if proba.shape[1] == 2:
                    oof_preds[val_idx, i] = proba[:, 1]
                else:
                    oof_preds[val_idx, i] = np.full(len(val_idx), proba[:, 0][0])
            models_per_label[i].append(clf)

    return oof_preds, models_per_label

# --- Main
if __name__ == "__main__":
    seed = 56
    np.random.seed(seed)

    engine = create_engine(
        "mssql+pyodbc://@localhost\\SQLSERV2019/NLP_DOGSTRUST?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    )

    df_all = pd.read_sql(
        """
        SELECT response_id, clean_text,
               lbl_01_survey, lbl_02_survey, lbl_03_survey, lbl_04_survey,
               lbl_01_gender, lbl_02_gender, lbl_03_gender,
               lbl_01_role, lbl_02_role, lbl_03_role,
               lbl_01_aid_type, lbl_02_aid_type, lbl_03_aid_type
        FROM NLP_DOGSTRUST.dbo.t_fullset_modelling_survey
        """,
        engine,
    )

    # Patch: merge gender 03 -> 02 at the dataframe level
    df_all = remap_gender_columns(df_all)

    texts = df_all["clean_text"].fillna("").tolist()
    response_ids = df_all["response_id"].tolist()

    # Embedding model
    model_st = SentenceTransformer("all-mpnet-base-v2", cache_folder="E:/huggingface")
    text_vecs = model_st.encode(texts, show_progress_bar=True)

    combo_all_correct: list[list[int]] = []
    final_results = df_all[["response_id"]].copy()
    preds_dict.clear()
    y_true_dict.clear()

    for task in LABEL_GROUPS:
        log(f"[TASK] Starting task: {task}")
        label_cols = LABEL_GROUPS[task]
        y_true = df_all[label_cols].values.astype(np.int32)

        # Load task keywords
        df_kw = pd.read_sql(
            f"""
            SELECT {CODE_COLUMN_MAP[task]} AS label_code, *
            FROM NLP_DOGSTRUST.dbo.{TABLE_MAP[task]}
            WHERE flag=1
            """,
            engine,
        )

        # Build robust keyword features (handles empty keyword tables and drops gender '03')
        X = safe_keyword_feature_matrix(model_st, text_vecs, df_kw, code_col="label_code", task=task)

        # CV to get thresholds/weights (logged & saved)
        thresholds, weights, aucs, threshold_map =run_cv_tuning_and_save(task, X, y_true, response_ids, engine)

        # Compute per-label class weights from data frequencies
        weights = []
        for i in range(y_true.shape[1]):
            counter = Counter(y_true[:, i])
            total = sum(counter.values())
            weight = {cls: total / (len(counter) * count) for cls, count in counter.items()}
            weights.append(weight)

        model_path = f"model_{task}_full.joblib"
        retrain = not os.path.exists(model_path)
        if retrain:
            model = train_per_label_rf(task, X, y_true, weights, seed, model_path)
        else:
            model = joblib.load(model_path)

        #probas = predict_proba_per_label(model, X)
        
        #preds = (probas >= 0.5).astype(int)
        probas = predict_proba_per_label(model, X)
        preds = apply_threshold(probas, threshold_map)  


        # One-hot argmax for single-choice tasks
        # One-hot argmax with threshold mask (for single-label tasks)
        if task == 'gender':
            # Ensure 2-label space only
            probas = probas[:, :2]
            y_true = y_true[:, :2]
            label_cols = label_cols[:2]

            # Build threshold mask
            thresholds = np.array([threshold_map.get(i, 0.5) for i in range(probas.shape[1])])
            mask = probas >= thresholds

            # Apply masked argmax
            probas_masked = np.where(mask, probas, 0)
            preds = np.zeros_like(probas, dtype=int)
            preds[np.arange(len(probas)), np.argmax(probas_masked, axis=1)] = 1

        elif task == 'role':
            thresholds = np.array([threshold_map.get(i, 0.5) for i in range(probas.shape[1])])
            mask = probas >= thresholds
            probas_masked = np.where(mask, probas, 0)
            preds = np.zeros_like(probas, dtype=int)
            preds[np.arange(len(probas)), np.argmax(probas_masked, axis=1)] = 1


        y_true_dict[task] = y_true
        preds_dict[task] = preds

        # Per-row correctness for hard combo
        correct_flags = []
        for i, rid in enumerate(response_ids):
            gt = y_true[i]
            pr = preds[i]
            score = int(np.all(gt == pr))
            correct_flags.append(score)

        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, average='macro')
        log(f"[SUMMARY] {task.upper()} — ACC: {acc:.3f} | F1: {f1:.3f}")
        save_single_task_scores(task, y_true, preds, engine)

        combo_all_correct.append(correct_flags)
        for j, col in enumerate(label_cols):
            final_results[f"pred_{col}"] = preds[:, j]
            final_results[f"true_{col}"] = y_true[:, j]
            final_results[f"proba_{col}"] = probas[:, j]

    combo_all_correct = np.array(combo_all_correct).T
    combo_hard = np.all(combo_all_correct == 1, axis=1)
    final_results["combo_hard_match"] = combo_hard
    log(f"[COMBO HARD MATCH] Accuracy: {combo_hard.mean():.3f} ({combo_hard.sum()}/{len(combo_hard)})")

    # Final overall metrics across all tasks
    y_true_all = final_results[[f"true_{col}" for cols in LABEL_GROUPS.values() for col in cols]].values
    y_pred_all = final_results[[f"pred_{col}" for cols in LABEL_GROUPS.values() for col in cols]].values
    log("=== FINAL MULTITASK EVALUATION ===")
    log(f"F1 macro: {f1_score(y_true_all, y_pred_all, average='macro', zero_division=0):.4f}")
    log(f"Accuracy: {accuracy_score(y_true_all, y_pred_all):.4f}")
    log(f"Hamming: {hamming_loss(y_true_all, y_pred_all):.4f}")
    log(f"Jaccard: {jaccard_score(y_true_all, y_pred_all, average='samples', zero_division=0):.4f}")
    log(f"Exact match: {np.all(y_true_all == y_pred_all, axis=1).mean():.4f}")
    n_mismatch = (~np.all(y_true_all == y_pred_all, axis=1)).sum()
    log(f"Total mismatch rows: {n_mismatch} of {len(y_true_all)}")

    final_results.to_sql("t_predictions_multitask", engine, schema="dbo", if_exists="replace", index=False)
    log("Saved predictions to t_predictions_multitask")

    # Exhaustive task-combination evaluation
    log("=== EXHAUSTIVE TASK COMBINATIONS EVALUATION ===")
    combo_results = []
    task_list = list(LABEL_GROUPS.keys())

    for r in range(1, len(task_list) + 1):
        for combo in combinations(task_list, r):
            metrics = evaluate_combo(preds_dict, y_true_dict, list(combo))
            combo_name = " + ".join(combo)
            log(f"[COMBO] {combo_name} | Match Acc={metrics['match_acc']:.4f} | Macro F1={metrics['macro_f1']:.4f} | ACC={metrics['macro_acc']:.4f}")
            combo_results.append({
                "task_combo": combo_name,
                "match_accuracy": metrics["match_acc"],
                "macro_f1": metrics["macro_f1"],
                "macro_accuracy": metrics["macro_acc"],
            })

    df_combo = pd.DataFrame(combo_results)
    df_combo.to_sql("t_combo_task_eval", engine, schema="dbo", if_exists="replace", index=False)
    log("Saved exhaustive task combinations to t_combo_task_eval")

    # Greedy forward stepwise task selection
    log("=== GREEDY STEPWISE TASK SELECTION ===")
    task_order: list[str] = []
    remaining = list(LABEL_GROUPS.keys())
    stepwise_metrics: list[dict] = []

    while remaining:
        best_score = -1.0
        best_task = None
        best_f1 = 0.0
        for task in remaining:
            test_tasks = task_order + [task]
            metrics = evaluate_combo(preds_dict, y_true_dict, test_tasks)
            match_acc = metrics['match_acc']
            macro_f1 = metrics['macro_f1']
            if match_acc > best_score:
                best_score = match_acc
                best_task = task
                best_f1 = macro_f1

        stepwise_metrics.append({
            'step': len(task_order) + 1,
            'task': best_task,
            'accuracy': best_score,
            'macro_f1': best_f1,
        })

        task_order.append(best_task)
        remaining.remove(best_task)
        log(f"Step {len(task_order)}: Add Task='{best_task}' | Acc={best_score:.4f} | F1={best_f1:.4f}")

    for s, task in enumerate(task_order, 1):
        subset_tasks = task_order[:s]
        metrics = evaluate_combo(preds_dict, y_true_dict, subset_tasks)
        acc = metrics['match_acc']
        f1 = metrics['macro_f1']

        # gains vs previous step
        prev_acc = stepwise_metrics[s - 2]['accuracy'] if s > 1 else 0.0
        prev_f1 = stepwise_metrics[s - 2]['macro_f1'] if s > 1 else 0.0
        stepwise_metrics[s - 1]['gain_acc'] = acc - prev_acc
        stepwise_metrics[s - 1]['gain_f1'] = f1 - prev_f1

        log(f"Step {s}: Tasks={subset_tasks} | Match Acc={acc:.4f} | F1={f1:.4f}")

    # Per-respondent stepwise match flags
    log("=== STEPWISE COMBO MATCH PER RESPONDENT ===")
    for i, rid in enumerate(final_results['response_id'].values):
        step_results = []
        for s in range(1, len(task_order) + 1):
            subset_tasks = task_order[:s]
            true_cols = [f"true_{col}" for t in subset_tasks for col in LABEL_GROUPS[t]]
            pred_cols = [f"pred_{col}" for t in subset_tasks for col in LABEL_GROUPS[t]]
            y_true_step = final_results[true_cols].values
            y_pred_step = final_results[pred_cols].values
            match = np.all(y_true_step == y_pred_step, axis=1)
            final_results[f"match_step_{s}"] = match
            step_results.append(int(match[i]))
        step_str = ' → '.join([f"{idx}:{val}" for idx, val in enumerate(step_results, 1)])
        log(f"RESP: {rid} | {step_str}")

    stepwise_cols = ['response_id'] + [f"match_step_{s}" for s in range(1, len(task_order) + 1)]
    stepwise_df = final_results[stepwise_cols].copy()
    stepwise_df.to_sql("t_stepwise_combo", engine, schema="dbo", if_exists="replace", index=False)
    log("Saved stepwise task evaluation to t_stepwise_combo")

    df_metrics = pd.DataFrame(stepwise_metrics)
    df_metrics.to_sql("t_stepwise_metrics", engine, schema="dbo", if_exists="replace", index=False)
    plot_stepwise_metrics_interactive(engine)
    log("Saved stepwise evaluation metrics to t_stepwise_metrics")
