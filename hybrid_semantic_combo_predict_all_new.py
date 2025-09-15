# file: hybrid_semantic_combo_predict_all_new.py

import time
import logging
import os
import argparse
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, accuracy_score, roc_curve, classification_report, hamming_loss, jaccard_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import joblib
from collections import Counter

logging.basicConfig(
    filename="combo_classifier_predict.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log(step):
    print(f"\033[92m[INFO]\033[0m {step}...")
    logging.info(step)

LABEL_GROUPS = {
    "method": ['lbl_01_survey', 'lbl_02_survey', 'lbl_03_survey', 'lbl_04_survey'],
    "gender": ['lbl_01_gender', 'lbl_02_gender', 'lbl_03_gender'],
    "role": ['lbl_01_role', 'lbl_02_role', 'lbl_03_role'],
    "aid_type": ['lbl_01_aid_type', 'lbl_02_aid_type', 'lbl_03_aid_type'],
}

TABLE_MAP = {
    "method": "t_keywords_by_methods",
    "role": "t_keywords_by_role",
    "gender": "t_keywords_by_gender",
    "aid_type": "t_keywords_by_aid_type"
}

CODE_COLUMN_MAP = {
    "method": "method_code",
    "role": "role_code",
    "gender": "gender_code",
    "aid_type": "aid_type_code"
}

MANUAL_WEIGHTS = {
    "method": [{0: 1.0, 1: 3.0}] * 4,
    "gender": [{0: 1.0, 1: 2.5}] * 3,
    "role": [{0: 1.0, 1: 4.0}] * 3,
    "aid_type": [{0: 1.0, 1: 3.0}] * 3,
}

parser = argparse.ArgumentParser()
parser.add_argument("--threshold-mode", choices=["manual", "auto"], default="auto")
parser.add_argument("--weight-mode", choices=["manual", "balanced"], default="balanced")
parser.add_argument("--n-estimators", type=int, default=10)
args = parser.parse_args()

THRESHOLD_MODE = args.threshold_mode
WEIGHT_MODE = args.weight_mode
N_ESTIMATORS = args.n_estimators
log(f"THRESHOLD MODE: {THRESHOLD_MODE.upper()} | WEIGHT MODE: {WEIGHT_MODE.upper()} | N_ESTIMATORS: {N_ESTIMATORS}")

def find_optimal_thresholds(y_true, probas):
    thresholds = []
    for i in range(y_true.shape[1]):
        y, p = y_true[:, i], probas[:, i]
        if len(np.unique(y)) < 2:
            thresholds.append(0.5)
            continue
        fpr, tpr, thresh = roc_curve(y, p)
        youden = tpr - fpr
        best = thresh[np.argmax(youden)]
        thresholds.append(best)
    return thresholds

def save_thresholds(task, thresholds):
    joblib.dump(thresholds, f"thresholds_{task}.joblib")

def apply_threshold_auto(probas, thresholds):
    preds = np.zeros_like(probas, dtype=int)
    for i in range(probas.shape[1]):
        t = thresholds[i] if i < len(thresholds) else 0.5
        preds[:, i] = (probas[:, i] >= t).astype(int)
    return preds

def get_class_weights(y_true, task):
    if WEIGHT_MODE == 'manual':
        if task not in MANUAL_WEIGHTS:
            raise ValueError(f"Manual weights not defined for task: {task}")
        return MANUAL_WEIGHTS[task]
    weights = []
    for i in range(y_true.shape[1]):
        counter = Counter(y_true[:, i])
        total = sum(counter.values())
        class_weight = {cls: total / (len(counter) * count) for cls, count in counter.items()}
        weights.append(class_weight)
        log(f"{task} - label {i} class weight: {class_weight} | dist: {dict(counter)}")
    return weights

def stack_positive_class_probas(moc: MultiOutputClassifier, probas_list, n_samples):
    cols = []
    for j, arr in enumerate(probas_list):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        try:
            classes = getattr(moc.estimators_[j], 'classes_', np.array([0, 1]))
        except Exception:
            classes = np.array([0, 1])

        if arr.shape[1] == 1:
            positive_prob = 1.0 if (len(classes) == 1 and classes[0] == 1) else 0.0
            col = np.full((n_samples,), positive_prob, dtype=float)
        else:
            if 1 in classes:
                idx = int(np.where(classes == 1)[0][0])
            else:
                idx = arr.shape[1] - 1
            col = arr[:, idx].astype(float)
        cols.append(col.reshape(-1, 1))
    return np.hstack(cols)

if __name__ == "__main__":
    engine = create_engine("mssql+pyodbc://@localhost\SQLSERV2019/NLP_DOGSTRUST?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes")
    df_all = pd.read_sql("""
        SELECT response_id, clean_text,
            lbl_01_survey, lbl_02_survey, lbl_03_survey, lbl_04_survey,
            lbl_01_gender, lbl_02_gender, lbl_03_gender,
            lbl_01_role, lbl_02_role, lbl_03_role,
            lbl_01_aid_type, lbl_02_aid_type, lbl_03_aid_type
        FROM NLP_DOGSTRUST.dbo.t_fullset_modelling_survey
    """, engine)

    texts = df_all['clean_text'].fillna('').tolist()
    response_ids = df_all['response_id'].tolist()
    model_st = SentenceTransformer("all-mpnet-base-v2", cache_folder="E:/huggingface")
    text_vecs = model_st.encode(texts, show_progress_bar=True)

    combo_all_correct = []
    final_results = df_all[['response_id']].copy()

    for task in LABEL_GROUPS:
        log(f"[TASK] Starting task: {task}")
        label_cols = LABEL_GROUPS[task]
        y_true = df_all[label_cols].values.astype(np.int32)

        if task == "method":
            df_kw = pd.read_sql(f"""
                SELECT {CODE_COLUMN_MAP[task]} AS label_code, keyword
                FROM NLP_DOGSTRUST.dbo.{TABLE_MAP[task]}
                Where (0.3 * semantic_score + 0.5 * semantic_prompt_score + 0.2 * jsd_score)>0.42 
            """, engine)
            log(f"[INFO] Loaded keywords for  method-Total: {len(df_kw)}")

        elif task == "role":
            df_kw = pd.read_sql(f"""
                SELECT {CODE_COLUMN_MAP[task]} AS label_code, keyword
                FROM NLP_DOGSTRUST.dbo.{TABLE_MAP[task]}
                WHERE flag=1
            """, engine)
            log(f"[INFO] Loaded keywords for role  - Total: {len(df_kw)}")

        elif task == "gender":
            df_kw = pd.read_sql(f"""
                SELECT {CODE_COLUMN_MAP[task]} AS label_code, keyword
                FROM NLP_DOGSTRUST.dbo.{TABLE_MAP[task]}
                WHERE flag=1 or score =0.95
            """, engine)
            log(f"[INFO] Loaded keywords for gender - Total: {len(df_kw)}")

        elif task == "aid_type":
            df_kw = pd.read_sql(f"""
                SELECT {CODE_COLUMN_MAP[task]} AS label_code, keyword
                FROM NLP_DOGSTRUST.dbo.{TABLE_MAP[task]}
                where flag=1
            """, engine)
            log(f"[INFO] Loaded keywords for aid_type  — Total: {len(df_kw)}")

        else:
            raise ValueError(f"Unknown task: {task}")

        label_codes = sorted(df_kw['label_code'].unique())
        kw_by_code = df_kw.groupby('label_code')


        keyword_vecs = {
            code: model_st.encode(kw_by_code.get_group(code)['keyword'].tolist(), show_progress_bar=False)
            for code in label_codes
        }
        max_kw = df_kw.groupby('label_code').size().max()
        log(f"[INFO] Max keyword per label: {max_kw}")

        keyword_features = []
        for vec in text_vecs:
            row = []
            for code in label_codes:
                sims = cosine_similarity([vec], keyword_vecs[code])[0]
                padded = np.pad(sims, (0, max_kw - len(sims)))[:max_kw]
                row.append(np.concatenate([padded, [np.mean(sims), np.max(sims), float(np.any(sims > 0.5))]]))
            keyword_features.append(np.concatenate(row))

        keyword_features = np.array(keyword_features, dtype=np.float32)
        X = keyword_features
        log(f"[INFO] Final feature shape: {X.shape}")

        weights = get_class_weights(y_true, task)
        model_path = f"model_{task}_full.joblib"
        retrain = False

        if os.path.exists(model_path):
            model = joblib.load(model_path)
            try:
                expected_features = model.estimators_[0].n_features_in_
                if expected_features != X.shape[1]:
                    log(f"[WARN] Feature mismatch: model expects {expected_features}, but got {X.shape[1]} — retraining model.")
                    retrain = True
                else:
                    log(f"[INFO] Loaded model with matching feature size: {expected_features}")
            except AttributeError:
                log("[WARN] Cannot verify model input feature size — retraining model.")
                retrain = True
        else:
            log(f"[INFO] No existing model found — training new model.")
            retrain = True

        if retrain:
            base_rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=42)
            model = MultiOutputClassifier(estimator=base_rf, n_jobs=-1)
            model.fit(X, y_true)
            for i, est in enumerate(model.estimators_):
                try:
                    est.set_params(class_weight=weights[i])
                except Exception:
                    pass
            joblib.dump(model, model_path)
            log(f"[INFO] Model trained and saved to {model_path}")

        raw_probas = model.predict_proba(X)
        probas = stack_positive_class_probas(model, raw_probas, X.shape[0])

        if THRESHOLD_MODE == "auto":
            thresholds = find_optimal_thresholds(y_true, probas)
            save_thresholds(task, thresholds)
        else:
            thresholds = [0.5] * y_true.shape[1]

        preds = apply_threshold_auto(probas, thresholds)

        correct_flags = []
        for i, rid in enumerate(response_ids):
            gt = y_true[i]
            pr = preds[i]
            correct = gt == pr
            score = int(correct.all())
            correct_flags.append(score)
            correct_str = ''.join(['✅' if x else '❌' for x in correct])
            log(f"[RESP: {rid}] {task}: {correct_str} ({int(correct.sum())}/{len(correct)}) | pred={pr.tolist()} | true={gt.tolist()}")

        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, average='macro')
        log(f"[SUMMARY] {task.upper()} — ACC: {acc:.3f} | F1: {f1:.3f}")

        combo_all_correct.append(correct_flags)
        for j, col in enumerate(label_cols):
            final_results[f"pred_{col}"] = preds[:, j]
            final_results[f"true_{col}"] = y_true[:, j]
            final_results[f"proba_{col}"] = probas[:, j]

    combo_all_correct = np.array(combo_all_correct).T
    combo_hard = np.all(combo_all_correct == 1, axis=1)
    final_results["combo_hard_match"] = combo_hard
    log(f"[COMBO HARD MATCH] Accuracy: {combo_hard.mean():.3f} ({combo_hard.sum()}/{len(combo_hard)})")

    y_true_all = final_results[[f"true_{col}" for cols in LABEL_GROUPS.values() for col in cols]].values
    y_pred_all = final_results[[f"pred_{col}" for cols in LABEL_GROUPS.values() for col in cols]].values
    log("=== FINAL MULTITASK EVALUATION ===")
    log(f"F1 macro: {f1_score(y_true_all, y_pred_all, average='macro', zero_division=0):.4f}")
    log(f"Accuracy: {accuracy_score(y_true_all, y_pred_all):.4f}")
    log(f"Hamming: {hamming_loss(y_true_all, y_pred_all):.4f}")
    log(f"Jaccard: {jaccard_score(y_true_all, y_pred_all, average='samples', zero_division=0):.4f}")
    log(f"Exact match: {np.all(y_true_all == y_pred_all, axis=1).mean():.4f}")

    final_results.to_sql("t_predictions_multitask", engine, schema="dbo", if_exists="replace", index=False)
    log("Saved predictions to t_predictions_multitask")
