# main.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

if __name__ == '__main__':
    data_path = r"breast_cancer_data.csv"  
    df = pd.read_csv(data_path)
    print("Data sample:")
    print(df.head(), "\n")
    print("Columns:", list(df.columns), "\n")


    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")] + ["id"]
    df = df.drop(columns=drop_cols, errors="ignore")


    if "diagnosis" in df.columns:
        y = df["diagnosis"].map({"M": 1, "B": 0})
        X = df.drop(columns=["diagnosis"])
    else:
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]


    X = pd.get_dummies(X, drop_first=True)

    for c in X.columns:
        if X[c].dtype.kind in "biufc":
            X[c] = X[c].fillna(X[c].median())

    if y.dtype == "O":
        y = y.astype("category").cat.codes

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"✅ Training complete. Test Accuracy: {acc:.3f}")

    os.makedirs("model", exist_ok=True)
    out_path = os.path.join("model", "breast_cancer_model.pkl")
    joblib.dump(model, out_path)
    print(f"💾 Model saved to: {out_path}")
