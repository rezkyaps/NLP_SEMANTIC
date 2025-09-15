import time
import random
import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, accuracy_score, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

random.seed(56)
np.random.seed(56)

logging.basicConfig(
    filename="method_classifier_cv.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log(msg):
    print(f"[INFO] {msg}")

def build_features_and_labels(df_survey, df_kw, model):
    for col in ['semantic_score', 'final_score', 'gate_factor', 'semantic_prompt_score']:
        df_kw[col] = df_kw[col].astype(str).str.replace(',', '.').astype(float)

    codes = sorted(df_kw['method_code'].unique())
    grouped_keywords = df_kw.groupby('method_code')

    keyword_vectors = {}
    score_map = {col: {} for col in ['semantic_score', 'final_score', 'gate_factor', 'semantic_prompt_score']}

    for code in codes:
        group = grouped_keywords.get_group(code)
        keyword_vectors[code] = model.encode(group['keyword'].tolist(), show_progress_bar=False)
        for col in score_map:
            score_map[col][code] = group[col].values

    texts = df_survey['clean_text'].fillna('').tolist()
    text_embeddings = model.encode(texts, show_progress_bar=True)

    MAX_KEYWORDS = df_kw.groupby('method_code').size().max()
    log(f"MAX_KEYWORDS set to: {MAX_KEYWORDS}...")

    semantic_features = []
    for text_emb in text_embeddings:
        row_feature = []
        for code in codes:
            kw_vecs = keyword_vectors[code]
            sims = cosine_similarity([text_emb], kw_vecs)[0]

            padded_features = []
            for col in ['semantic_score', 'final_score', 'gate_factor', 'semantic_prompt_score']:
                weighted = sims * score_map[col][code]
                padded = np.pad(weighted, (0, MAX_KEYWORDS - len(weighted))) if len(weighted) < MAX_KEYWORDS else weighted[:MAX_KEYWORDS]
                padded_features.append(padded)

            mean_sim = np.mean(sims)
            max_sim = np.max(sims)
            presence_flag = np.any(sims > 0.5).astype(float)

            full_feature = np.concatenate(padded_features + [[mean_sim, max_sim, presence_flag]])
            row_feature.append(full_feature)

        semantic_features.append(np.concatenate(row_feature))

    X = np.array(semantic_features).astype(np.float32)
    y = df_survey.filter(like='lbl_').values.astype(np.int32)
    return X, y, text_embeddings, df_survey


def compute_weights(y):
    weights = []
    for i in range(y.shape[1]):
        unique_vals = np.unique(y[:, i])
        if len(unique_vals) == 1:
            weights.append(1.0)
        else:
            w = compute_class_weight(class_weight='balanced', classes=unique_vals, y=y[:, i])
            weights.append(w[1])
        log(f"Weight for label_{i}: {weights[-1]:.2f}")
    return weights

def guided_oversample(X, y, df_survey=None, min_pos=15):
    X_aug, y_aug = X.copy(), y.copy()
    df_aug = df_survey.copy() if df_survey is not None else None
    for i in range(y.shape[1]):
        pos_idx = np.where(y[:, i] == 1)[0]
        needed = max(min_pos - len(pos_idx), 0)
        log(f"Before oversample label {i}: pos={len(pos_idx)}, needed={needed}")
        if needed > 0 and len(pos_idx) > 0:
            sampled_idx = np.random.choice(pos_idx, size=needed, replace=True)
            X_new = X[sampled_idx]
            y_new = y[sampled_idx].copy()
            X_aug = np.vstack([X_aug, X_new])
            y_aug = np.vstack([y_aug, y_new])
            if df_aug is not None:
                df_aug = pd.concat([df_aug, df_aug.iloc[sampled_idx]], ignore_index=True)
        log(f"\u2705 After oversample label {i}: pos={np.sum(y_aug[:, i]==1)}, neg={np.sum(y_aug[:, i]==0)}")
    return X_aug, y_aug, df_aug

def apply_threshold(probas, threshold_map=None, default_threshold=0.5):
    preds = np.zeros_like(probas, dtype=int)
    for i in range(probas.shape[1]):
        threshold = threshold_map.get(i, default_threshold) if threshold_map else default_threshold
        preds[:, i] = (probas[:, i] >= threshold).astype(int)
    return preds

def find_best_threshold(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return 1.0, None, None, None
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden_index = tpr - fpr
    best_thresh = thresholds[np.argmax(youden_index)]
    return best_thresh, fpr, tpr, thresholds

def evaluate_and_log_all(X, y, best_estimators, threshold_map, df_survey, default_threshold=None):
    probas = np.column_stack([
        clf.predict_proba(X)[:, 1] if len(clf.classes_) > 1 else np.zeros(X.shape[0])
        for clf in best_estimators
    ])
    preds = apply_threshold(probas, threshold_map, default_threshold)
    f1 = f1_score(y, preds, average='macro', zero_division=0)
    acc = accuracy_score(y, preds)
    log(f"\n===== FULL TRAIN PREDICTION RESULTS =====")
    log(f"F1-score: {f1:.4f} | Accuracy: {acc:.4f}")
    for i in range(len(preds)):
        rid = df_survey.iloc[i]['response_id']
        log(f"[FULL] Respondent ID: {rid} | Pred: {preds[i].tolist()} | True: {y[i].tolist()}")

def robustness_split_and_evaluate(X_os, y_os, df_survey_os, best_estimators, threshold_map, default_threshold=None, folds=5):
    log("\n===== ROBUSTNESS TEST (split on oversampled data) =====")
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=56)
    stratify_label = y_os[:, -1].astype(int)

    f1_list, acc_list = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_os, stratify_label)):
        X_test = X_os[test_idx]
        y_test = y_os[test_idx]
        df_test = df_survey_os.iloc[test_idx].reset_index(drop=True)

        probas = np.column_stack([
            clf.predict_proba(X_test)[:, 1] if len(clf.classes_) > 1 else np.zeros(X_test.shape[0])
            for clf in best_estimators
        ])
        preds = apply_threshold(probas, threshold_map, default_threshold)
        f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        acc = accuracy_score(y_test, preds)

        f1_list.append(f1)
        acc_list.append(acc)

        log(f"\n[ROBUST-FOLD-{fold+1}] Macro F1: {f1:.4f} | Accuracy: {acc:.4f}")
        for i in range(len(preds)):
            rid = df_test.iloc[i]['response_id']
            log(f"[ROBUST-FOLD-{fold+1}] Respondent ID: {rid} | Pred: {preds[i].tolist()} | True: {y_test[i].tolist()}")

    log("\n===== ROBUSTNESS FOLD AVERAGE SCORES =====")
    log(f"Average Macro F1: {np.mean(f1_list):.4f}")
    log(f"Average Accuracy: {np.mean(acc_list):.4f}")

def crossval_with_fixed_threshold(X, y, df_survey, use_tuned_threshold=True, folds=5):
    seed = 56
    log(f"\n===== Running CV evaluation using global weights & thresholds (seed={seed}) =====")

    X_os, y_os, df_os = guided_oversample(X, y, df_survey, min_pos=15)
    global_weights = compute_weights(y_os)
    threshold_map = {}
    best_estimators = []

    for i in range(y.shape[1]):
        clf = RandomForestClassifier(
            class_weight={0: 1.0, 1: global_weights[i]},
            random_state=seed,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=6,
            max_features='sqrt',
            n_estimators=20
        )
        clf.fit(X_os, y_os[:, i])
        best_estimators.append(clf)

        y_prob = clf.predict_proba(X)[:, 1] if len(clf.classes_) == 2 else np.zeros(X.shape[0])
        best_thresh, fpr, tpr, _ = find_best_threshold(y[:, i], y_prob)
        threshold_map[i] = best_thresh
        auc_val = auc(fpr, tpr) if fpr is not None else float('nan')

        if use_tuned_threshold:
            log(f"[GLOBAL] Label {i}: Threshold = {best_thresh:.3f}, AUC = {auc_val:.3f}")
        else:
            log(f"[GLOBAL] Label {i}: Threshold = {manual_thresholds[i]:.3f} (manual override)")

    manual_thresholds = {i: 0.5 for i in range(y.shape[1])}
    evaluate_and_log_all(X, y, best_estimators, threshold_map if use_tuned_threshold else manual_thresholds, df_survey, default_threshold=0.5)
    robustness_split_and_evaluate(X_os, y_os, df_os, best_estimators, threshold_map if use_tuned_threshold else manual_thresholds, default_threshold=0.5, folds=folds)

if __name__ == "__main__":
    start_time = time.time()
    seed = 56
    np.random.seed(seed)
    conn_str = "mssql+pyodbc://@localhost\\SQLSERV2019/NLP_DOGSTRUST_DEPLOY?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
    engine = create_engine(conn_str)

    log("Loading data from SQL")
    df_survey = pd.read_sql("""
        SELECT response_id, clean_text,
               lbl_01_survey, lbl_02_survey, lbl_03_survey, lbl_04_survey
        FROM NLP_DOGSTRUST.dbo.t_fullset_modelling_survey
    """, engine)

    df_kw = pd.read_sql("""
        SELECT *
        FROM NLP_DOGSTRUST.dbo.t_key_report_method
        WHERE flag=1
    """, engine)

    log("Loading transformer model")
    model = SentenceTransformer("all-mpnet-base-v2", cache_folder="E:/huggingface")

    X, y, _, df_survey_full = build_features_and_labels(df_survey, df_kw, model)
    log(f"Final feature matrix shape: {X.shape}...")
    log(f"Final label matrix shape: {y.shape}...")

    crossval_with_fixed_threshold(X, y, df_survey_full)
    log(f"Pipeline completed in {time.time() - start_time:.2f} seconds.")
