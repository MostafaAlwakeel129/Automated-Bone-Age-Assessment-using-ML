from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np


def train_random_forest(X_train, y_train, rf_config):
    """Original function — unchanged."""
    print(f"\n Training Random Forest model")
    model = RandomForestRegressor(
        n_estimators=rf_config['n_estimators'],
        max_depth=rf_config['max_depth'],
        min_samples_split=rf_config['min_samples_split'],
        random_state=rf_config['random_state'],
        n_jobs=rf_config['n_jobs']
    )
    model.fit(X_train, y_train)
    print(" Training complete.")
    return model


def tune_random_forest(X_train, y_train, rf_config):
    """Enhancement 1: Hyperparameter tuning via Grid Search."""
    print("\n Running hyperparameter tuning (Grid Search)...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
    }
    base_model = RandomForestRegressor(
        random_state=rf_config['random_state'],
        n_jobs=rf_config['n_jobs']
    )
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        verbose=2,
        n_jobs=rf_config['n_jobs']
    )
    grid_search.fit(X_train, y_train)
    print(f"\n Best parameters found: {grid_search.best_params_}")
    print(f" Best CV MAE: {-grid_search.best_score_:.2f} months")
    return grid_search.best_estimator_


def tune_gradient_boosting(X_train, y_train, rf_config):
    """
    Enhancement 3: Gradient Boosting with tuning.
    Builds trees sequentially — each tree corrects the errors of the previous one.
    Usually outperforms Random Forest on tabular data.
    Smaller grid than RF search to keep runtime reasonable.
    """
    print("\n Running Gradient Boosting tuning...")

    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [4, 6],
        'learning_rate': [0.05, 0.1],
        'min_samples_split': [2, 5],
    }

    base_model = GradientBoostingRegressor(
        random_state=rf_config['random_state']
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        verbose=2,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print(f"\n Best parameters found: {grid_search.best_params_}")
    print(f" Best CV MAE: {-grid_search.best_score_:.2f} months")
    return grid_search.best_estimator_


def train_gender_split(X_train, y_train, X_test, y_test, rf_config):
    """Enhancement 2: Gender-split models using Gradient Boosting."""
    print("\n Training gender-split models...")

    male_train_mask   = X_train[:, -1] > 0
    female_train_mask = X_train[:, -1] <= 0
    male_test_mask    = X_test[:, -1] > 0
    female_test_mask  = X_test[:, -1] <= 0

    X_train_male,   y_train_male   = X_train[male_train_mask],   y_train[male_train_mask]
    X_train_female, y_train_female = X_train[female_train_mask], y_train[female_train_mask]
    X_test_male,    y_test_male    = X_test[male_test_mask],     y_test[male_test_mask]
    X_test_female,  y_test_female  = X_test[female_test_mask],   y_test[female_test_mask]

    print(f"   Male samples  — train: {len(X_train_male)}, test: {len(X_test_male)}")
    print(f"   Female samples — train: {len(X_train_female)}, test: {len(X_test_female)}")

    print("\n Tuning male model (Gradient Boosting)...")
    male_model = tune_gradient_boosting(X_train_male, y_train_male, rf_config)

    print("\n Tuning female model (Gradient Boosting)...")
    female_model = tune_gradient_boosting(X_train_female, y_train_female, rf_config)

    return (male_model,   X_test_male,   y_test_male,
            female_model, X_test_female, y_test_female)