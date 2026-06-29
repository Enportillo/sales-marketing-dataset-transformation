import json
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from src.Routes import RUTAS


@dataclass
class ExperimentResult:
    feature_set: str
    roc_auc: float
    avg_precision: float
    best_params: dict


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    eps = 1e-6

    # Proxy de antiguedad cuando no existe signup_date.
    out["estimated_days_since_signup"] = (
        (out["lifetime_value"] / (out["avg_order_value"] + eps)) * 30.0
    ).clip(lower=1.0, upper=3650.0)

    out["purchase_frequency_intensity"] = (
        out["last_3_month_purchase_freq"] / (out["estimated_days_since_signup"] + eps)
    ) * 30.0

    out["ticket_promedio_estimado"] = out["lifetime_value"] / (
        out["last_3_month_purchase_freq"] + 1.0
    )

    out["marketing_efficiency"] = out["lifetime_value"] / (
        out["marketing_spend_per_user"] + 1.0
    )

    out["friction_free_customer"] = (
        (out["support_tickets"] == 0)
        & (out["delivery_delay_days"] == 0)
    ).astype(int)

    out["engagement_score"] = (
        out["total_visits"] * out["pages_per_session"] * out["avg_session_time"]
    )

    out["email_engagement"] = out["email_open_rate"] * out["email_click_rate"]

    out["nps_satisfaction_gap"] = out["nps_score"] - (out["satisfaction_score"] * 20.0)

    out["spend_per_visit"] = out["total_spent"] / (out["total_visits"] + 1.0)
    out["value_per_ticket"] = out["lifetime_value"] / (out["support_tickets"] + 1.0)

    return out


def run_search(X_train: pd.DataFrame, y_train: pd.Series) -> RandomizedSearchCV:
    model = RandomForestClassifier(random_state=42, n_jobs=-1)

    param_distributions = {
        "n_estimators": [200, 300, 500, 700, 900],
        "max_depth": [None, 8, 12, 16, 24, 32],
        "min_samples_split": [2, 4, 8, 12, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.5, 0.7, None],
        "class_weight": [None, "balanced", "balanced_subsample"],
        "criterion": ["gini", "entropy", "log_loss"],
        "bootstrap": [True],
        "max_samples": [None, 0.6, 0.8, 0.9],
        "ccp_alpha": [0.0, 1e-4, 1e-3, 5e-3],
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=12,
        cv=4,
        scoring={"roc_auc": "roc_auc", "avg_precision": "average_precision"},
        refit="roc_auc",
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search


def run_fast_screening(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> tuple[float, float]:
    model = RandomForestClassifier(
        random_state=42,
        n_estimators=400,
        max_depth=14,
        min_samples_split=6,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, probs)
    ap = average_precision_score(y_test, probs)
    return float(roc), float(ap)


def main() -> None:
    path = RUTAS["data_processed"] / "Sales_Marketing_Clean_(Codificado).csv"
    df = pd.read_csv(path)

    target = "subscription_type"
    y = df[target].astype(int)
    X_base = df.drop(columns=[target]).copy()
    X_eng = add_engineered_features(X_base)

    behavior_cols = [
        "total_spent",
        "avg_order_value",
        "last_3_month_purchase_freq",
        "total_visits",
        "pages_per_session",
        "support_tickets",
    ]

    v4_cols = behavior_cols

    engineered_cols = [
        "estimated_days_since_signup",
        "purchase_frequency_intensity",
        "ticket_promedio_estimado",
        "marketing_efficiency",
        "friction_free_customer",
        "engagement_score",
        "email_engagement",
        "nps_satisfaction_gap",
        "spend_per_visit",
        "value_per_ticket",
    ]

    candidate_sets = {
        "v4_baseline_behavior_only": v4_cols,
        "v5_behavior_plus_core4": v4_cols
        + [
            "purchase_frequency_intensity",
            "ticket_promedio_estimado",
            "marketing_efficiency",
            "friction_free_customer",
        ],
        "v5_behavior_plus_all_engineered": v4_cols + engineered_cols,
        "v5_all_original_plus_engineered": list(X_base.columns) + engineered_cols,
        "v5_business_compact": [
            "lifetime_value",
            "total_spent",
            "avg_order_value",
            "last_3_month_purchase_freq",
            "marketing_spend_per_user",
            "support_tickets",
            "delivery_delay_days",
            "nps_score",
            "satisfaction_score",
            "purchase_frequency_intensity",
            "ticket_promedio_estimado",
            "marketing_efficiency",
            "friction_free_customer",
            "value_per_ticket",
        ],
    }

    # Probar tambien combinaciones de engineered sobre base de comportamiento.
    for k in [2, 3, 4]:
        for combo in combinations(engineered_cols[:6], k):
            key = f"v5_behavior_combo_{'_'.join(combo)}"
            candidate_sets[key] = v4_cols + list(combo)

    # Fase 1: screening rapido para quedarnos con los candidatos mas prometedores.
    screening_rows = []
    for i, (name, cols) in enumerate(candidate_sets.items(), start=1):
        X = X_eng[cols].copy()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        roc, ap = run_fast_screening(X_train, y_train, X_test, y_test)
        screening_rows.append((name, cols, roc, ap))
        print(f"[screening {i}/{len(candidate_sets)}] {name}: ROC-AUC={roc:.4f} | AP={ap:.4f}")

    screening_rows.sort(key=lambda row: (row[2], row[3]), reverse=True)
    top_candidates = screening_rows[:5]

    print("\nTop candidatos tras screening:")
    for row in top_candidates:
        print(f"- {row[0]}: ROC-AUC={row[2]:.4f} | AP={row[3]:.4f}")

    # Fase 2: tuning profundo solo en los mejores candidatos.
    results: list[ExperimentResult] = []

    for name, cols, _, _ in top_candidates:
        X = X_eng[cols].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        search = run_search(X_train, y_train)
        best_model = search.best_estimator_

        probs = best_model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, probs)
        ap = average_precision_score(y_test, probs)

        results.append(
            ExperimentResult(
                feature_set=name,
                roc_auc=float(roc),
                avg_precision=float(ap),
                best_params=search.best_params_,
            )
        )
        print(f"[tuning] {name}: ROC-AUC={roc:.4f} | AP={ap:.4f}")

    results_sorted = sorted(results, key=lambda r: (r.roc_auc, r.avg_precision), reverse=True)

    print("\nTop 5 resultados:")
    for row in results_sorted[:5]:
        print(
            f"- {row.feature_set}: ROC-AUC={row.roc_auc:.4f} | "
            f"AP={row.avg_precision:.4f}"
        )

    output_path = RUTAS["reports"] / "v5_feature_search_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "feature_set": r.feature_set,
                    "roc_auc": r.roc_auc,
                    "avg_precision": r.avg_precision,
                    "best_params": r.best_params,
                }
                for r in results_sorted
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nResultados guardados en: {output_path}")


if __name__ == "__main__":
    main()
