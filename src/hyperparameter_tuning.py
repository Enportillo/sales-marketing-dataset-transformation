from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def optimize_random_forest(X_train, y_train):

    pipeline = Pipeline([
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 5, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2'],
        'classifier__class_weight': [None, 'balanced']
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2
    )

    print("Buscando mejores hiperparámetros...")
    grid_search.fit(X_train, y_train)

    print("\nMejores parámetros:")
    print(grid_search.best_params_)

    print(f"\nMejor ROC-AUC CV: {grid_search.best_score_:.4f}")

    return grid_search


def optimize_random_forest_v2(X_train, y_train):

    pipeline = Pipeline([
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [100, 150],
        'classifier__max_depth': [3, 5, 8, 12],
        'classifier__min_samples_split': [5, 10, 20],
        'classifier__min_samples_leaf': [2, 4, 6],
        'classifier__max_features': ['sqrt'],
        'classifier__class_weight': ['balanced', 'balanced_subsample']
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2
    )

    print("Buscando mejores hiperparámetros para v2...")
    grid_search.fit(X_train, y_train)

    print("\nMejores parámetros v2:")
    print(grid_search.best_params_)

    print(f"\nMejor ROC-AUC CV v2: {grid_search.best_score_:.4f}")

    return grid_search


def optimize_random_forest_v3(X_train, y_train):

    pipeline = Pipeline([
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [500, 600],
        'classifier__max_depth': [None, 12],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2],
        'classifier__max_features': ['sqrt', 0.5, 4],
        'classifier__class_weight': ['balanced', 'balanced_subsample']
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2
    )

    print("Buscando mejores hiperparámetros para v3 orientados a upselling...")
    grid_search.fit(X_train, y_train)

    print("\nMejores parámetros v3:")
    print(grid_search.best_params_)

    print(f"\nMejor ROC-AUC CV v3: {grid_search.best_score_:.4f}")

    return grid_search


def optimize_random_forest_v3_fast(X_train, y_train):

    pipeline = Pipeline([
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Version ligera para iterar rapido en notebook.
    param_grid = {
        'classifier__n_estimators': [120, 200],
        'classifier__max_depth': [10, None],
        'classifier__min_samples_split': [2, 10],
        'classifier__min_samples_leaf': [1, 2],
        'classifier__max_features': ['sqrt', 0.5],
        'classifier__class_weight': ['balanced_subsample']
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2
    )

    print("Buscando mejores hiperparametros para v3 fast...")
    grid_search.fit(X_train, y_train)

    print("\nMejores parametros v3 fast:")
    print(grid_search.best_params_)

    print(f"\nMejor ROC-AUC CV v3 fast: {grid_search.best_score_:.4f}")

    return grid_search


def optimize_random_forest_v4(X_train, y_train):

    pipeline = Pipeline([
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    # Exploracion orientada a mejorar ranking de clientes con potencial de upgrade.
    param_distributions = {
        'classifier__n_estimators': [200, 300, 400, 600, 800],
        'classifier__max_depth': [None, 10, 15, 20, 30],
        'classifier__min_samples_split': [2, 5, 10, 20],
        'classifier__min_samples_leaf': [1, 2, 4, 8],
        'classifier__max_features': ['sqrt', 'log2', 0.5, 0.7, None],
        'classifier__class_weight': [None, 'balanced', 'balanced_subsample'],
        'classifier__criterion': ['gini', 'entropy', 'log_loss'],
        'classifier__bootstrap': [True],
        'classifier__max_samples': [None, 0.7, 0.85],
        'classifier__ccp_alpha': [0.0, 0.0001, 0.001, 0.005]
    }

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=30,
        cv=4,
        scoring={
            'roc_auc': 'roc_auc',
            'avg_precision': 'average_precision'
        },
        refit='avg_precision',
        n_jobs=-1,
        random_state=42,
        verbose=2
    )

    print("Buscando mejores hiperparametros para v4 orientados a upselling...")
    random_search.fit(X_train, y_train)

    print("\nMejores parametros v4:")
    print(random_search.best_params_)

    best_idx = random_search.best_index_
    best_roc_auc = random_search.cv_results_['mean_test_roc_auc'][best_idx]

    print(f"\nMejor Average Precision CV v4: {random_search.best_score_:.4f}")
    print(f"Mejor ROC-AUC CV v4 (mismo modelo): {best_roc_auc:.4f}")

    return random_search


def build_v5_engineered_features(X):

    X_v5 = X.copy()
    eps = 1e-6

    # Proxy de antiguedad: se aproxima con relacion entre valor historico y ticket promedio.
    X_v5['estimated_days_since_signup'] = (
        (X_v5['lifetime_value'] / (X_v5['avg_order_value'] + eps)) * 30.0
    ).clip(lower=1.0, upper=3650.0)

    X_v5['purchase_frequency_intensity'] = (
        X_v5['last_3_month_purchase_freq'] / (X_v5['estimated_days_since_signup'] + eps)
    ) * 30.0

    X_v5['ticket_promedio_estimado'] = (
        X_v5['lifetime_value'] / (X_v5['last_3_month_purchase_freq'] + 1.0)
    )

    X_v5['marketing_efficiency'] = (
        X_v5['lifetime_value'] / (X_v5['marketing_spend_per_user'] + 1.0)
    )

    X_v5['friction_free_customer'] = (
        (X_v5['support_tickets'] == 0)
        & (X_v5['delivery_delay_days'] == 0)
    ).astype(int)

    X_v5['engagement_score'] = (
        X_v5['total_visits'] * X_v5['pages_per_session'] * X_v5['avg_session_time']
    )

    X_v5['email_engagement'] = X_v5['email_open_rate'] * X_v5['email_click_rate']
    X_v5['spend_per_visit'] = X_v5['total_spent'] / (X_v5['total_visits'] + 1.0)
    X_v5['value_per_ticket'] = X_v5['lifetime_value'] / (X_v5['support_tickets'] + 1.0)

    return X_v5


def get_v5_feature_sets(X):

    base_cols = [
        'total_spent',
        'avg_order_value',
        'last_3_month_purchase_freq',
        'total_visits',
        'pages_per_session',
        'support_tickets',
    ]

    engineered_cols = [
        'estimated_days_since_signup',
        'purchase_frequency_intensity',
        'ticket_promedio_estimado',
        'marketing_efficiency',
        'friction_free_customer',
        'engagement_score',
        'email_engagement',
        'spend_per_visit',
        'value_per_ticket',
    ]

    feature_sets = {
        'v4_baseline_behavior': base_cols,
        'v5_core_business': base_cols + [
            'purchase_frequency_intensity',
            'ticket_promedio_estimado',
            'marketing_efficiency',
            'friction_free_customer',
        ],
        'v5_best_screening': base_cols + [
            'purchase_frequency_intensity',
            'ticket_promedio_estimado',
        ],
        'v5_full_engineering': base_cols + engineered_cols,
        'v5_all_original_plus_engineered': list(X.columns) + engineered_cols,
    }

    return feature_sets


def optimize_random_forest_v5(X_train, y_train):

    pipeline = Pipeline([
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

    param_distributions = {
        'classifier__n_estimators': [200, 300, 400, 600, 800],
        'classifier__max_depth': [None, 10, 15, 20, 30],
        'classifier__min_samples_split': [2, 5, 10, 20],
        'classifier__min_samples_leaf': [1, 2, 4, 8],
        'classifier__max_features': ['sqrt', 'log2', 0.5, 0.7, None],
        'classifier__class_weight': [None, 'balanced', 'balanced_subsample'],
        'classifier__criterion': ['gini', 'entropy', 'log_loss'],
        'classifier__bootstrap': [True],
        'classifier__max_samples': [None, 0.7, 0.85],
        'classifier__ccp_alpha': [0.0, 0.0001, 0.001, 0.005]
    }

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=30,
        cv=4,
        scoring={
            'roc_auc': 'roc_auc',
            'avg_precision': 'average_precision'
        },
        refit='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=2
    )

    print("Buscando mejores hiperparametros para v5 con feature engineering...")
    random_search.fit(X_train, y_train)

    print("\nMejores parametros v5:")
    print(random_search.best_params_)

    best_idx = random_search.best_index_
    best_ap = random_search.cv_results_['mean_test_avg_precision'][best_idx]

    print(f"\nMejor ROC-AUC CV v5: {random_search.best_score_:.4f}")
    print(f"Mejor Average Precision CV v5 (mismo modelo): {best_ap:.4f}")

    return random_search