import os
import time
import logging
import joblib
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from sqlalchemy import create_engine
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report, hamming_loss, jaccard_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.multioutput import MultiOutputClassifier
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from imblearn.over_sampling import RandomOverSampler
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = lambda msg: print(f"\033[92m[INFO]\033[0m {msg}")

conn_str = "mssql+pyodbc://@localhost\\SQLSERV2019/NLP_DOGSTRUST?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
engine = create_engine(conn_str)

LABEL_GROUPS = {
    "method": ['lbl_01_survey', 'lbl_02_survey', 'lbl_03_survey', 'lbl_04_survey'],
    "role": ['lbl_01_role', 'lbl_02_role', 'lbl_03_role'],
    "gender": ['lbl_01_gender', 'lbl_02_gender', 'lbl_03_gender'],
    "aid_type": ['lbl_01_aid_type', 'lbl_02_aid_type', 'lbl_03_aid_type'],
}

KEYWORD_TABLES = {
    "method": ("t_keywords_by_methods", "method_code"),
    "role": ("t_keywords_by_role", "role_code"),
    "gender": ("t_keywords_by_gender", "gender_code"),
    "aid_type": ("t_keywords_by_aid_type", "aid_type_code"),
}

MODEL = SentenceTransformer("all-mpnet-base-v2", cache_folder="E:/huggingface")

def build_features(text_vecs, df_kw, label_col):
    df_kw['score'] = df_kw['score'].astype(str).str.replace(',', '.').astype(float)
    label_codes = sorted(df_kw[label_col].unique())
    kw_by_label = df_kw.groupby(label_col)

    vectors, scores = {}, {}
    for code in label_codes:
        group = kw_by_label.get_group(code)
        vectors[code] = MODEL.encode(group['keyword'].tolist(), show_progress_bar=False)
        scores[code] = group['score'].values

    MAX_KW = df_kw.groupby(label_col).size().max()
    log(f"MAX_KEYWORDS label_code: {MAX_KW}")

    feats = []
    for vec in text_vecs:
        row_feat = []
        for code in label_codes:
            sims = cosine_similarity([vec], vectors[code])[0]
            weighted = sims * scores[code]
            mean_sim, max_sim = np.mean(sims), np.max(sims)
            present = float(np.any(sims > 0.5))
            padded = np.pad(weighted, (0, MAX_KW - len(weighted)))[:MAX_KW]
            row_feat.append(np.concatenate([padded, [mean_sim, max_sim, present]]))
        feats.append(np.concatenate(row_feat))
    return np.array(feats, dtype=np.float32)

def oversample_only_gender_label03(task, X, y):
    if task != "gender":
        return X, y
    col_idx = 2
    counts = {int(k): int(v) for k, v in zip(*np.unique(y[:, col_idx], return_counts=True))}
    if counts.get(1, 0) > 1:
        return X, y
    log("Applying oversample only on gender label 03")
    ros = RandomOverSampler()
    X_os, y_os = ros.fit_resample(X, y[:, col_idx])
    needed = 9 - counts.get(1, 0)
    add_idx = np.where(y_os == 1)[0][-needed:]
    X_new = X_os[add_idx]
    y_new = np.zeros((len(add_idx), y.shape[1]), dtype=int)
    y_new[:, col_idx] = 1
    X_aug = np.vstack([X, X_new])
    y_aug = np.vstack([y, y_new])
    return X_aug, y_aug

def compute_weights(y):
    weights = []
    for i in range(y.shape[1]):
        uniq = np.unique(y[:, i])
        w = compute_class_weight(class_weight='balanced', classes=uniq, y=y[:, i]) if len(uniq) > 1 else [1.0]
        weight = w[1] if len(w) > 1 else 1.0
        log(f"Weight for label_{i}: {weight:.2f}")
        weights.append(weight)
    return weights

def train_all(task):
    label_cols = LABEL_GROUPS[task]
    table, code_col = KEYWORD_TABLES[task]
    log(f"\nTrain task: {task}")
    df_kw = pd.read_sql(f"SELECT {code_col} AS label_code, keyword, score FROM NLP_DOGSTRUST.dbo.{table}", engine)
    df = pd.read_sql("SELECT response_id, clean_text, " + ", ".join(label_cols) + " FROM NLP_DOGSTRUST.dbo.t_fullset_modelling_survey", engine)
    df = df.dropna(subset=['clean_text'])

    texts = df['clean_text'].tolist()
    y = df[label_cols].values.astype(np.int32)

    for i, col in enumerate(label_cols):
        counts = {int(k): int(v) for k, v in zip(*np.unique(y[:, i], return_counts=True))}
        log(f"{col} label counts: {counts}")

    log("Encoding features...")
    X = build_features(MODEL.encode(texts, show_progress_bar=True), df_kw, 'label_code')

    log("Evaluating model on full data (predict all)...")
    full_weights = compute_weights(y)
    models = [RandomForestClassifier(n_estimators=5, class_weight={0: 1.0, 1: full_weights[i]}, random_state=56) for i in range(y.shape[1])]
    full_model = MultiOutputClassifier(models[0])
    full_model.estimators_ = models
    full_model.fit(X, y)
    y_full_pred = full_model.predict(X)

    print(f"\n\033[94m=== Full Training Evaluation Results - {task} ===\033[0m")
    print(classification_report(y, y_full_pred, target_names=label_cols, zero_division=0))
    print("F1 macro:", round(f1_score(y, y_full_pred, average='macro', zero_division=0), 4))
    print("Accuracy:", round(accuracy_score(y, y_full_pred), 4))
    print("Hamming:", round(hamming_loss(y, y_full_pred), 4))
    print("Jaccard:", round(jaccard_score(y, y_full_pred, average='samples', zero_division=0), 4))
    print("Exact match:", round(np.all(y == y_full_pred, axis=1).mean(), 4))

    df_results = df[['response_id']].copy()
    df_results[[f"pred_{col}" for col in label_cols]] = y_full_pred
    df_results[[f"true_{col}" for col in label_cols]] = y
    df_results.to_csv(f"full_predict_results_{task}.csv", index=False)

    log("Splitting 20% holdout test from full data...")
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=56)
    for train_idx, test_idx in msss.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

    X_train, y_train = oversample_only_gender_label03(task, X_train, y_train)
    weights = full_weights

    log("Tuning n_estimators for Random Forest...")
    best_estimators = []
    for i in range(y.shape[1]):
        rf = RandomForestClassifier(class_weight={0: 1.0, 1: weights[i]}, random_state=56)
        grid = GridSearchCV(
            rf,
            param_grid={"n_estimators": [5, 6, 7, 8]},
            scoring="f1_macro",
            cv=3,
            n_jobs=-1
        )
        grid.fit(X_train, y_train[:, i])
        best_estimators.append(grid.best_estimator_)
        log(f"Best n_estimators for label_{i}: {grid.best_params_['n_estimators']}")

    clf = MultiOutputClassifier(best_estimators[0])
    clf.estimators_ = best_estimators
    clf.fit(X_train, y_train)
    joblib.dump(clf, f"model_{task}_full.joblib")

    log("Evaluating trained model on holdout 20% test data...")
    y_pred = clf.predict(X_test)

    print(f"\n\033[94m=== Holdout Test Evaluation Results - {task} ===\033[0m")
    print(classification_report(y_test, y_pred, target_names=label_cols, zero_division=0))
    print("F1 macro:", round(f1_score(y_test, y_pred, average='macro', zero_division=0), 4))
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("Hamming:", round(hamming_loss(y_test, y_pred), 4))
    print("Jaccard:", round(jaccard_score(y_test, y_pred, average='samples', zero_division=0), 4))
    print("Exact match:", round(np.all(y_test == y_pred, axis=1).mean(), 4))

    df_holdout = df.iloc[test_idx].copy()
    df_holdout[[f"pred_{col}" for col in label_cols]] = y_pred
    df_holdout[[f"true_{col}" for col in label_cols]] = y_test
    df_holdout.to_csv(f"holdout_predict_results_{task}.csv", index=False)

    return clf, df, label_cols, df_kw


def main():
    df_all = pd.read_sql("SELECT response_id, clean_text, " + ", ".join([c for cols in LABEL_GROUPS.values() for c in cols]) + " FROM NLP_DOGSTRUST.dbo.t_fullset_modelling_survey", engine)
    df_all = df_all.dropna(subset=['clean_text'])
    texts = df_all['clean_text'].tolist()
    result_df = df_all[['response_id', 'clean_text']].copy()

    for task in LABEL_GROUPS:
        model, df_task, label_cols, df_kw = train_all(task)
        log(f"Running predict all on full data for task: {task}")
        X = build_features(MODEL.encode(texts, show_progress_bar=True), df_kw, 'label_code')
        y_pred = model.predict(X)
        for i, col in enumerate(label_cols):
            result_df[f"pred_{col}"] = y_pred[:, i]

        print(f"\n\033[94m=== Predict All - {task} ===\033[0m")
        y_true = df_all[label_cols].values.astype(np.int32)
        print(classification_report(y_true, y_pred, target_names=label_cols, zero_division=0))
        print("F1 macro:", round(f1_score(y_true, y_pred, average='macro', zero_division=0), 4))
        print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
        print("Hamming:", round(hamming_loss(y_true, y_pred), 4))
        print("Jaccard:", round(jaccard_score(y_true, y_pred, average='samples', zero_division=0), 4))
        print("Exact match:", round(np.all(y_true == y_pred, axis=1).mean(), 4))

    print("\n=== FINAL MULTITASK EVALUATION ===")
    true_cols = [c for cols in LABEL_GROUPS.values() for c in cols]
    pred_cols = [f"pred_{c}" for c in true_cols]
    y_true_all = df_all[true_cols].values
    y_pred_all = result_df[pred_cols].values
    print("F1 macro:", round(f1_score(y_true_all, y_pred_all, average='macro', zero_division=0), 4))
    print("Accuracy:", round(accuracy_score(y_true_all, y_pred_all), 4))
    print("Hamming:", round(hamming_loss(y_true_all, y_pred_all), 4))
    print("Jaccard:", round(jaccard_score(y_true_all, y_pred_all, average='samples', zero_division=0), 4))
    print("Exact match:", round(np.all(y_true_all == y_pred_all, axis=1).mean(), 4))

    result_df.to_sql("t_predictions_multitask", engine, schema="dbo", if_exists="replace", index=False)
    log("Saved predictions to t_predictions_multitask")

if __name__ == "__main__":
    start = time.time()
    main()
    log(f"Complete in {time.time() - start:.2f}s")
