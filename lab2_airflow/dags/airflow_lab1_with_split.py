# dags/airflow_lab1_full.py
from datetime import datetime, timedelta
import os
import pickle
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

try:
    from kneed import KneeLocator
    KNEED_AVAILABLE = True
except Exception:
    KNEED_AVAILABLE = False

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# ------------------ Paths ------------------
BASE_DIR   = "/opt/airflow/working_data"
DATA_DIR   = f"{BASE_DIR}/data"
TMP_DIR    = f"{BASE_DIR}/tmp"
MODEL_DIR  = f"{BASE_DIR}/model"
MODEL_PATH = f"{MODEL_DIR}/kmeans_model.pkl"
SSE_PATH   = f"{TMP_DIR}/sse.json"     # small JSON with SSE values
PROC_PATH  = f"{TMP_DIR}/preproc.pkl"  # fitted preprocessor pipeline

# Your dataset on host: .\dags\data\CollegePlacement.csv
INPUT_FILE = "/opt/airflow/dags/data/CollegePlacement.csv"

# Ensure directories exist
for d in (DATA_DIR, TMP_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)

# ------------------ Helpers ------------------
def split_dataset(input_csv: str,
                  test_size: float = 0.2,
                  random_state: int = 42) -> dict:
    """
    Split CSV into train/test and return their paths (tiny XCom).
    Works for ANY tabular dataset: no hard-coded columns.
    """
    df = pd.read_csv(input_csv)
    # basic cleaning: drop empty columns
    df = df.loc[:, ~df.columns.str.match(r'^\s*$')]
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path  = os.path.join(DATA_DIR, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"[split] train={len(train_df)} -> {train_path}")
    print(f"[split] test ={len(test_df)} -> {test_path}")
    return {"train": train_path, "test": test_path}

def build_preprocessor(train_csv: str) -> str:
    """
    Fit a preprocessing pipeline on the TRAIN split:
      - One-hot encode categoricals
      - Standardize numerics
      - Drop rows that end up all-NaN after transformation
    Persist the fitted pipeline to PROC_PATH and return that path.
    """
    df = pd.read_csv(train_csv)

    # Identify column types
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    # If there are no numeric columns, force at least one-hot to produce features
    num_trans = "passthrough" if numeric_cols else "drop"
    cat_trans = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    if numeric_cols:
        from sklearn.impute import SimpleImputer
        num_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
    else:
        num_pipe = "drop"

    preproc = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_trans, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Fit the preprocessor
    X = preproc.fit_transform(df)
    # Remove rows that became NaN-only (shouldn’t happen, but be safe)
    if np.isnan(X).any():
        # Replace remaining NaNs with column means
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])

    with open(PROC_PATH, "wb") as f:
        pickle.dump(preproc, f)
    print(f"[preproc] fitted pipeline → {PROC_PATH}  | X shape after transform: {X.shape}")
    return PROC_PATH

def train_kmeans(preproc_path: str,
                 train_csv: str,
                 model_path: str,
                 sse_path: str,
                 k_min: int = 1,
                 k_max: int = 10) -> dict:
    """
    Train KMeans for k in [k_min, k_max] on preprocessed TRAIN data.
    Save the best model (by elbow) and SSE list. Return metadata.
    """
    with open(preproc_path, "rb") as f:
        preproc = pickle.load(f)

    df = pd.read_csv(train_csv)
    X = preproc.transform(df)
    # guard again for any NaNs
    if np.isnan(X).any():
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])

    # Fit models
    sse = []
    ks = list(range(k_min, k_max + 1))
    for k in ks:
        # n_init="auto" works in newer sklearn; fall back to int if needed
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        except TypeError:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        sse.append(float(km.inertia_))

    # Determine elbow
    if KNEED_AVAILABLE and len(ks) >= 3:
        kl = KneeLocator(ks, sse, curve="convex", direction="decreasing")
        k_best = int(kl.elbow) if kl.elbow else int(np.argmin(np.gradient(np.gradient(sse))) + k_min)
    else:
        # Heuristic: pick k with largest relative drop in SSE
        drops = [ (sse[i-1] - sse[i]) / max(sse[i-1], 1e-9) for i in range(1, len(sse)) ]
        k_best = ks[int(np.argmax(drops) + 1)] if drops else ks[0]

    # Train final model at k_best
    try:
        final_km = KMeans(n_clusters=k_best, random_state=42, n_init="auto")
    except TypeError:
        final_km = KMeans(n_clusters=k_best, random_state=42, n_init=10)
    final_km.fit(X)

    # Save model and SSE
    with open(model_path, "wb") as f:
        pickle.dump({"model": final_km, "k": k_best}, f)
    with open(sse_path, "w") as f:
        json.dump({"k_range": ks, "sse": sse, "k_best": k_best}, f)

    print(f"[train] saved model → {model_path}  | k_best={k_best}")
    print(f"[train] SSE list saved → {sse_path}")
    return {"model_path": model_path, "sse_path": sse_path, "k_best": k_best, "n_samples": int(X.shape[0]), "n_features": int(X.shape[1])}

def elbow_report(model_path: str, sse_path: str) -> str:
    """
    Load artifacts and emit a short, human-readable summary string.
    Keep return tiny for XCom.
    """
    with open(model_path, "rb") as f:
        payload = pickle.load(f)
    with open(sse_path, "r") as f:
        sse_obj = json.load(f)
    summary = f"k_best={payload['k']} | samples={sse_obj['k_range'][0]}..{sse_obj['k_range'][-1]} tried"
    print(f"[elbow] {summary}")
    return summary

# ------------------ DAG ------------------
default_args = {
    "owner": "your_name",
    "start_date": datetime(2025, 1, 15),
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="Airflow_Lab1_Full",
    description="Split → preprocess → train KMeans → elbow → notify (self-contained, file-based)",
    default_args=default_args,
    schedule=None,   # manual trigger
    catchup=False,
    tags=["lab1", "ml", "kmeans", "file-handoff"],
) as dag:

    split_task = PythonOperator(
        task_id="split_dataset_task",
        python_callable=split_dataset,
        op_kwargs={"input_csv": INPUT_FILE, "test_size": 0.2, "random_state": 42},
    )

    preproc_task = PythonOperator(
        task_id="fit_preprocessor_task",
        python_callable=build_preprocessor,
        op_kwargs={"train_csv": "{{ ti.xcom_pull('split_dataset_task')['train'] }}"},
    )

    train_task = PythonOperator(
        task_id="train_kmeans_task",
        python_callable=train_kmeans,
        op_kwargs={
            "preproc_path": preproc_task.output,
            "train_csv": "{{ ti.xcom_pull('split_dataset_task')['train'] }}",
            "model_path": MODEL_PATH,
            "sse_path": SSE_PATH,
            "k_min": 1,
            "k_max": 10,
        },
    )

    elbow_task = PythonOperator(
        task_id="elbow_report_task",
        python_callable=elbow_report,
        op_kwargs={"model_path": MODEL_PATH, "sse_path": SSE_PATH},
    )

    notify_task = BashOperator(
        task_id="notify_done",
        bash_command=(
            'echo "✅ Pipeline completed successfully!" && '
            f'echo "Model: {MODEL_PATH}" && '
            f'echo "SSE  : {SSE_PATH}" && '
            'echo "Elbow : {{ ti.xcom_pull(task_ids=\'elbow_report_task\') }}" && '
            'echo "Timestamp: $(date -u +\"%Y-%m-%dT%H:%M:%SZ\")"'
        ),
    )

    split_task >> preproc_task >> train_task >> elbow_task >> notify_task

if __name__ == "__main__":
    dag.test()
