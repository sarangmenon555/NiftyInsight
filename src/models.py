import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from . import config, features

CORE_COLUMNS = ["ret_126", "vol_63", "dist_sma_200", "dist_52w_high"]
MODEL_NAMES = ["dt", "rf", "xgb"]

DEFAULTS = {
    "dt": {"max_depth": 6, "min_samples_leaf": 500},
    "rf": {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 300, "max_features": "sqrt"},
    "xgb": {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 100},
}

SPACES = {
    "dt": {
        "max_depth": [3, 4, 5, 6, 8],
        "min_samples_leaf": [200, 500, 1000, 2000],
        "criterion": ["gini", "entropy"],
    },
    "rf": {
        "n_estimators": [150, 250],
        "max_depth": [8, 10, 12],
        "min_samples_leaf": [100, 300, 600],
        "max_features": ["sqrt", 0.3],
    },
    "xgb": {
        "n_estimators": [200, 400, 600],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.9],
        "colsample_bytree": [0.5, 0.8],
        "min_child_weight": [20, 100, 300],
    },
}

FAST_SPACES = {
    "dt": SPACES["dt"],
    "rf": {"n_estimators": [40, 80], "max_depth": [6, 8], "min_samples_leaf": [100, 300], "max_features": ["sqrt"]},
    "xgb": {"n_estimators": [60, 120], "max_depth": [3, 4], "learning_rate": [0.05, 0.1], "subsample": [0.8], "colsample_bytree": [0.7], "min_child_weight": [50, 200]},
}

CONTEXT_COLUMNS = ["date", "symbol", "sector", "price", "target_cls", "target_reg", "ret_fwd", "vol_21", "log_turnover", "flag_regime_ok"]


def make_model(name, params=None):
    params = dict(DEFAULTS[name] if params is None else params)
    if name == "dt":
        return DecisionTreeClassifier(random_state=config.RANDOM_STATE, **params)
    if name == "rf":
        return RandomForestClassifier(n_jobs=-1, random_state=config.RANDOM_STATE, **params)
    if name == "xgb":
        return XGBClassifier(tree_method="hist", n_jobs=-1, random_state=config.RANDOM_STATE, eval_metric="logloss", **params)
    raise ValueError(name)


def time_splits(df):
    train_end = pd.Timestamp(config.TRAIN_END)
    valid_end = pd.Timestamp(config.VALID_END)
    purge = pd.Timedelta(days=config.PURGE_DAYS)
    usable = df["target_cls"].notna() & df[CORE_COLUMNS].notna().all(axis=1)
    train = usable & (df["date"] <= train_end - purge)
    valid = usable & (df["date"] > train_end) & (df["date"] <= valid_end - purge)
    test = usable & (df["date"] > valid_end)
    return {"train": train.to_numpy(), "valid": valid.to_numpy(), "test": test.to_numpy(), "usable": usable.to_numpy()}


def take(df, mask, columns, limit, seed):
    index = np.flatnonzero(mask)
    if limit is not None and len(index) > limit:
        index = np.sort(np.random.default_rng(seed).choice(index, limit, replace=False))
    subset = df.iloc[index]
    return subset[columns].to_numpy(dtype="float32"), subset["target_cls"].to_numpy(dtype="int8")


def tune_model(name, train_xy, valid_xy, n_iter, fast):
    x_train, y_train = train_xy
    x_valid, y_valid = valid_xy
    x = np.vstack([x_train, x_valid])
    y = np.concatenate([y_train, y_valid])
    fold = np.concatenate([np.full(len(y_train), -1), np.zeros(len(y_valid))])
    space = (FAST_SPACES if fast else SPACES)[name]
    combos = int(np.prod([len(v) for v in space.values()]))
    search = RandomizedSearchCV(
        make_model(name, {}),
        space,
        n_iter=min(n_iter, combos),
        scoring="average_precision",
        cv=PredefinedSplit(fold),
        refit=False,
        random_state=config.RANDOM_STATE,
        n_jobs=1,
    )
    search.fit(x, y)
    return dict(search.best_params_), float(search.best_score_)


def evaluate(y_true, proba):
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    top = max(1, int(len(proba) * 0.01))
    order = np.argsort(-proba)[:top]
    return {
        "rows": int(len(y_true)),
        "base_rate": float(y_true.mean()),
        "auc": float(roc_auc_score(y_true, proba)),
        "average_precision": float(average_precision_score(y_true, proba)),
        "precision_top_1pct": float(y_true[order].mean()),
    }


def choose_thresholds(preds):
    valid = preds[preds["split"] == "valid"]
    round_trip = 2.0 * config.COST_PER_SIDE
    thresholds = {}
    for column in config.PREDICTION_COLUMNS:
        best = None
        for quantile in np.arange(0.90, 0.996, 0.005):
            threshold = float(valid[column].quantile(quantile))
            selected = valid.loc[valid[column] >= threshold, "ret_fwd"] - round_trip
            if len(selected) < 300 or selected.std() == 0:
                continue
            tstat = float(selected.mean() / (selected.std() / np.sqrt(len(selected))))
            if best is None or tstat > best["tstat"]:
                best = {
                    "threshold": threshold,
                    "quantile": float(quantile),
                    "valid_rows_selected": int(len(selected)),
                    "valid_mean_net_return": float(selected.mean()),
                    "tstat": tstat,
                }
        if best is None:
            threshold = float(valid[column].quantile(0.95))
            best = {"threshold": threshold, "quantile": 0.95, "valid_rows_selected": 0, "valid_mean_net_return": float("nan"), "tstat": float("nan")}
        thresholds[column] = best
    return thresholds


def importance_table(models, columns):
    frames = []
    for name in ("rf", "xgb"):
        if name in models:
            values = models[name].feature_importances_
            frames.append(pd.DataFrame({"feature": columns, "importance": values, "model": name}))
    table = pd.concat(frames, ignore_index=True)
    return table.sort_values(["model", "importance"], ascending=[True, False]).reset_index(drop=True)


def train_all(df, fast=False, final=True):
    config.ensure_dirs()
    columns = features.feature_columns(df)
    splits = time_splits(df)
    limits = {
        "tune_train": 40_000 if fast else 200_000,
        "tune_valid": 20_000 if fast else 100_000,
        "fit_train": 100_000 if fast else 600_000,
        "final": 150_000 if fast else 900_000,
        "n_iter": 3 if fast else 10,
    }
    if splits["train"].sum() == 0 or splits["valid"].sum() == 0 or splits["test"].sum() == 0:
        raise RuntimeError("Time splits are empty. Check ALGO_START_DATE, ALGO_TRAIN_END and ALGO_VALID_END against the data range.")

    tune_train = take(df, splits["train"], columns, limits["tune_train"], 1)
    tune_valid = take(df, splits["valid"], columns, limits["tune_valid"], 2)
    best_params, tuning = {}, {}
    for name in MODEL_NAMES:
        params, score = tune_model(name, tune_train, tune_valid, limits["n_iter"], fast)
        best_params[name] = params
        tuning[name] = score

    x_train, y_train = take(df, splits["train"], columns, limits["fit_train"], 3)
    fitted = {}
    for name in MODEL_NAMES:
        model = make_model(name, best_params[name])
        model.fit(x_train, y_train)
        fitted[name] = model

    eval_mask = splits["valid"] | splits["test"]
    eval_frame = df.loc[eval_mask, CONTEXT_COLUMNS].copy()
    eval_frame["split"] = np.where(splits["valid"][eval_mask], "valid", "test")
    x_eval = df.loc[eval_mask, columns].to_numpy(dtype="float32")
    for name in MODEL_NAMES:
        eval_frame[f"pred_{name}"] = fitted[name].predict_proba(x_eval)[:, 1].astype("float32")
    eval_frame["pred_ens"] = ((eval_frame["pred_rf"] + eval_frame["pred_xgb"]) / 2.0).astype("float32")
    eval_frame = eval_frame.reset_index(drop=True)

    rows = []
    for split in ("valid", "test"):
        part = eval_frame[eval_frame["split"] == split]
        for column in config.PREDICTION_COLUMNS:
            rows.append({"split": split, "prediction": column, **evaluate(part["target_cls"], part[column])})
    metrics = pd.DataFrame(rows)

    thresholds = choose_thresholds(eval_frame)
    importance = importance_table(fitted, columns)

    eval_frame.to_parquet(config.PREDICTIONS_PATH, index=False)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(config.REPORT_DIR / "model_metrics.csv", index=False)
    importance.to_csv(config.REPORT_DIR / "feature_importance.csv", index=False)
    config.BEST_PARAMS_PATH.write_text(json.dumps({"params": best_params, "tuning_average_precision": tuning}, indent=2))
    config.THRESHOLDS_PATH.write_text(json.dumps(thresholds, indent=2))
    config.FEATURE_COLUMNS_PATH.write_text(json.dumps(columns))

    if final:
        fit_final_models(df, columns, splits["usable"], best_params, limits["final"])
    return {"metrics": metrics, "thresholds": thresholds, "best_params": best_params, "importance": importance, "predictions": eval_frame}


def fit_final_models(df, columns, usable, best_params, limit):
    x, y = take(df, usable, columns, limit, 4)
    for name in MODEL_NAMES:
        model = make_model(name, best_params[name])
        model.fit(x, y)
        joblib.dump(model, config.MODEL_DIR / f"{name}_final.joblib")
    meta = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_end": str(df["date"].max().date()),
        "labeled_rows_used": int(len(y)),
    }
    config.MODEL_META_PATH.write_text(json.dumps(meta, indent=2))


def models_exist():
    return all((config.MODEL_DIR / f"{name}_final.joblib").exists() for name in MODEL_NAMES) and config.THRESHOLDS_PATH.exists()


def models_age_days():
    if not config.MODEL_META_PATH.exists():
        return None
    meta = json.loads(config.MODEL_META_PATH.read_text())
    trained = datetime.fromisoformat(meta["trained_at_utc"])
    return (datetime.now(timezone.utc) - trained).days


def score_frame(day_frame):
    columns = json.loads(config.FEATURE_COLUMNS_PATH.read_text())
    aligned = day_frame.reindex(columns=columns, fill_value=0)
    x = aligned.to_numpy(dtype="float32")
    scored = {}
    for name in MODEL_NAMES:
        model = joblib.load(config.MODEL_DIR / f"{name}_final.joblib")
        scored[f"pred_{name}"] = model.predict_proba(x)[:, 1]
    scored["pred_ens"] = (scored["pred_rf"] + scored["pred_xgb"]) / 2.0
    return pd.DataFrame(scored, index=day_frame.index)
