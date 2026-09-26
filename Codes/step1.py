"""Step 1: validate the single 918-record dataset and freeze the internal split.

Run directly from the Code directory: python step1.py
The Excel input has NO HEADER. No input record is ever changed on disk.
"""

import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.metaestimators import available_if


DATA_FILE = Path(os.environ.get(
    "HEART_DATA_FILE",
    r"D:\data\heart_unique_records_no_header.xlsx",
))
RESULTS = Path(os.environ.get("HEART_RESULTS_DIR", str(DATA_FILE.parent / "reanalysis_918")))
COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "target"]
NUM = ["age", "trestbps", "chol", "thalach", "oldpeak"]
NOMINAL = ["cp", "restecg", "slope"]
BINARY = ["sex", "fbs", "exang"]
SEED = 42
SPLIT_FRACTION = 0.40


def load_data():
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Put the headerless workbook here: {DATA_FILE}")
    data = pd.read_excel(DATA_FILE, header=None, engine="openpyxl")
    if data.shape != (918, 12):
        raise ValueError(f"Expected 918 data rows and 12 columns, found {data.shape}. No header row is skipped.")
    data.columns = COLS
    for col in COLS:
        data[col] = pd.to_numeric(data[col], errors="raise")
    if data.target.isna().any() or not data.target.isin((0, 1)).all():
        raise ValueError("target must consist only of 0 (absent) and 1 (present)")
    for col, accepted in {"sex": (0, 1), "cp": (1, 2, 3, 4), "fbs": (0, 1),
                          "restecg": (0, 1, 2), "exang": (0, 1),
                          "slope": (0, 1, 2, 3)}.items():
        if not data[col].dropna().isin(accepted).all():
            raise ValueError(f"Unexpected category in {col}; verify the source codebook")
    if data.duplicated(COLS).any():
        raise ValueError(f"Found {data.duplicated(COLS).sum()} exact duplicate rows")
    if data.duplicated(COLS[:-1]).any():
        raise ValueError("Predictor-identical records present; check target conflicts and source provenance")
    if data.target.value_counts().to_dict() != {1: 508, 0: 410}:
        raise ValueError("Unexpected class distribution; verify this is the audited 918-record workbook")
    return data


def data_fingerprint():
    return hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()


def manifest():
    path = RESULTS / "analysis_manifest.json"
    if not path.exists():
        raise FileNotFoundError("Run step1.py before steps 2 and 3")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["data_sha256"] != data_fingerprint():
        raise ValueError("Dataset changed since step 1: do not mix results from different datasets")
    return record


def split_data(data):
    development, test = train_test_split(np.arange(len(data)), test_size=SPLIT_FRACTION,
                                         stratify=data.target, random_state=SEED)
    if not set(development).isdisjoint(test):
        raise AssertionError("Split overlap")
    return (data.iloc[development][COLS[:-1]].copy(),
            data.iloc[test][COLS[:-1]].copy(),
            data.iloc[development].target.astype(int).copy(),
            data.iloc[test].target.astype(int).copy())


def cleaned_features(data):
    """Treat physiologically impossible zeros as missing, only in memory."""
    frame = data[COLS[:-1]].copy()
    frame.loc[frame.chol.eq(0), "chol"] = np.nan
    frame.loc[frame.trestbps.eq(0), "trestbps"] = np.nan
    # The IEEE source contains one slope=0. Keep it as its own category.
    return frame


class ClinicalEstimator(ClassifierMixin, BaseEstimator):
    """Impute, augment TRAINING observations, encode, and fit a classifier.

    These operations occur inside fit(), which GridSearchCV calls afresh for
    every training fold. Validation and holdout observations are never augmented.
    """

    def __init__(self, model=None, encoding="onehot", ratio=0.5,
                 noise_multiplier=1.0, seed=SEED):
        self.model = model
        self.encoding = encoding
        self.ratio = ratio
        self.noise_multiplier = noise_multiplier
        self.seed = seed

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Supply a DataFrame with the 11 named predictor columns")
        X = X[COLS[:-1]]
        if self.model is None:
            raise ValueError("A model must be provided")
        self.numeric_imputer_ = SimpleImputer(strategy="median")
        self.categorical_imputer_ = SimpleImputer(strategy="most_frequent")
        training = self._impute(X, fit=True)
        y_values = np.asarray(y, dtype=int)
        rng = np.random.default_rng(self.seed)
        n = int(len(training) * self.ratio)
        if n:
            sampled = rng.choice(len(training), n, replace=True)
            extra = training.iloc[sampled].copy().reset_index(drop=True)
            extra["age"] = (extra.age + rng.choice([-1, 0, 1], n)).clip(18, 100)
            for col, sd, lower, upper in (("chol", 5, 80, 700),
                                           ("trestbps", 3, 70, 250),
                                           ("thalach", 3, 50, 230),
                                           ("oldpeak", .05, 0, 10)):
                extra[col] = (extra[col] + rng.normal(0, sd*self.noise_multiplier, n)).clip(lower, upper)
            training = pd.concat([training, extra], ignore_index=True)
            y_values = np.concatenate([y_values, y_values[sampled]])
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        if self.encoding == "onehot":
            pieces = [("numeric", StandardScaler(), NUM),
                      ("nominal", encoder, NOMINAL),
                      ("binary", "passthrough", BINARY)]
        elif self.encoding == "integer":
            pieces = [("numeric", StandardScaler(), NUM),
                      ("categorical", "passthrough", NOMINAL+BINARY)]
        else:
            raise ValueError("encoding must be 'onehot' or 'integer'")
        self.preprocessor_ = ColumnTransformer(pieces, verbose_feature_names_out=False)
        self.model_ = clone(self.model)
        self.model_.fit(self.preprocessor_.fit_transform(training), y_values)
        self.classes_ = self.model_.classes_
        return self

    def _impute(self, X, fit=False):
        frame = X[COLS[:-1]].copy()
        for cols, transformer in ((NUM, self.numeric_imputer_),
                                  (NOMINAL+BINARY, self.categorical_imputer_)):
            frame[cols] = (transformer.fit_transform(frame[cols]) if fit
                           else transformer.transform(frame[cols]))
        return frame

    def _prepared(self, X):
        return self.preprocessor_.transform(self._impute(X))

    def predict(self, X):
        return self.model_.predict(self._prepared(X))

    @available_if(lambda self: self.model is not None and hasattr(self.model, "predict_proba"))
    def predict_proba(self, X):
        return self.model_.predict_proba(self._prepared(X))

    @available_if(lambda self: self.model is not None and hasattr(self.model, "decision_function"))
    def decision_function(self, X):
        return self.model_.decision_function(self._prepared(X))


def model_specs():
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    try:
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("For all eight classifiers install: pip install pandas openpyxl scikit-learn scipy matplotlib lightgbm xgboost") from exc
    return {
        "Logistic Regression": (LogisticRegression(max_iter=5000, random_state=SEED), {"model__C": [.1, 1, 10]}),
        "KNN": (KNeighborsClassifier(), {"model__n_neighbors": [5, 9], "model__weights": ["uniform", "distance"]}),
        "SVM": (SVC(probability=True, random_state=SEED), {"model__C": [.1, 1, 10], "model__kernel": ["linear", "rbf"]}),
        "Decision Tree": (DecisionTreeClassifier(random_state=SEED), {"model__max_depth": [3, 5, None], "model__min_samples_leaf": [1, 4]}),
        "Random Forest": (RandomForestClassifier(random_state=SEED, n_jobs=1), {"model__n_estimators": [150, 300], "model__max_depth": [5, None], "model__min_samples_leaf": [1, 4]}),
        "Ridge Classifier": (RidgeClassifier(), {"model__alpha": [.1, 1, 10]}),
        "LightGBM": (LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=1), {"model__n_estimators": [100, 250], "model__learning_rate": [.03, .1], "model__num_leaves": [15, 31]}),
        "XGBoost": (XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=1, tree_method="hist"), {"model__n_estimators": [100, 250], "model__learning_rate": [.03, .1], "model__max_depth": [3, 5]}),
    }


def compact_repeated_grids():
    from sklearn.model_selection import ParameterGrid
    specs = model_specs()
    return {name: ({key: (vals[:1] if name in ("Random Forest", "LightGBM", "XGBoost") and key == "model__n_estimators" else vals)
                    for key, vals in grid.items()}) for name, (_, grid) in specs.items()}


def save_table(frame, filename):
    RESULTS.mkdir(parents=True, exist_ok=True)
    final = RESULTS / filename
    temp = RESULTS / (filename + ".tmp.xlsx")
    frame.to_excel(temp, index=False)
    os.replace(temp, final)


def main():
    raw = load_data()
    if RESULTS.exists() and (RESULTS / "analysis_manifest.json").exists():
        if manifest()["data_sha256"] != data_fingerprint():
            raise ValueError("Existing results belong to another dataset")
    RESULTS.mkdir(parents=True, exist_ok=True)
    data = raw.copy()
    data[COLS[:-1]] = cleaned_features(raw)
    audit = pd.DataFrame([{"Rows": len(raw), "Exact_duplicate_rows": int(raw.duplicated().sum()),
                            "Negative": int((raw.target==0).sum()), "Positive": int((raw.target==1).sum()),
                            "Zero_chol_treated_missing": int((raw.chol==0).sum()),
                            "Zero_trestbps_treated_missing": int((raw.trestbps==0).sum()),
                            "Slope_zero_kept_as_category": int((raw.slope==0).sum()),
                            "Input_SHA256": data_fingerprint()}])
    save_table(audit, "data_integrity.xlsx")
    dev, test = train_test_split(np.arange(len(data)), test_size=SPLIT_FRACTION,
                                 stratify=data.target, random_state=SEED)
    membership = pd.DataFrame({"Input_row_1_based": np.arange(1, len(data)+1),
                               "Target": data.target.astype(int),
                               "Partition": ["development" if i in set(dev) else "holdout" for i in range(len(data))]})
    save_table(membership, "split_membership.xlsx")
    rows = []
    for name in NUM:
        for target, group in data.groupby("target"):
            values = group[name].dropna()
            rows.append({"Feature": name, "Target": target, "N_measured": len(values),
                         "Mean": values.mean(), "SD": values.std(), "Median": values.median()})
    save_table(pd.DataFrame(rows), "eda_numeric_summary_by_target.xlsx")
    cat_rows = []
    for name in NOMINAL+BINARY:
        for (val, target), n in data.groupby([name,"target"],dropna=False).size().items():
            cat_rows.append({"Feature":name,"Category":val,"Target":target,"N":int(n)})
    save_table(pd.DataFrame(cat_rows),"eda_categorical_counts.xlsx")
    fig, axes = plt.subplots(2,3,figsize=(13,8))
    for ax,name in zip(axes.flat,NOMINAL+BINARY):
        pd.crosstab(data[name], data.target).plot.bar(ax=ax,legend=False,title=name)
    axes.flat[0].legend(title="Heart disease")
    fig.tight_layout(); fig.savefig(RESULTS/"figure1_categorical_features.png",dpi=180);plt.close(fig)
    fig, axes = plt.subplots(2,3,figsize=(13,8))
    for ax,name in zip(axes.flat,NUM):
        for target in (0,1):
            data.loc[data.target.eq(target),name].dropna().plot.hist(ax=ax,alpha=.45,bins=20,label=f"Target {target}")
        ax.set_title(name)
    axes.flat[0].legend();axes.flat[-1].axis("off")
    fig.tight_layout();fig.savefig(RESULTS/"figure2_numerical_features.png",dpi=180);plt.close(fig)
    params={"data_sha256":data_fingerprint(),"rows":918,"columns":COLS,
            "random_seed":SEED,"holdout_fraction":SPLIT_FRACTION,
            "development_n":len(dev),"holdout_n":len(test),
            "holdout_kind":"internal","feature_slope_zero":"kept as separate source category",
            "source_wise_validation":"not possible from one combined data source"}
    (RESULTS/"analysis_manifest.json").write_text(json.dumps(params,indent=2),encoding="utf-8")
    print(f"Step 1 completed: {len(dev)} development; {len(test)} internal holdout; results: {RESULTS}",flush=True)


if __name__ == "__main__":
    main()
