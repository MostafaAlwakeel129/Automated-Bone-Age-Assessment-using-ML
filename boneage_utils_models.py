from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

def train_random_forest(X_train, y_train, rf_config):

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

def train_xgboost(X_train, y_train, xgb_config):
    print("\n Training XGBoost model with Grid Search...")

    xgb_model = XGBRegressor(
        random_state=xgb_config['random_state'],
        n_jobs=xgb_config['n_jobs']
    )

    param_grid = {
        'learning_rate': xgb_config['learning_rate'],
        'max_depth':     xgb_config['max_depth'],
        'n_estimators':  xgb_config['n_estimators'],
        'subsample':     xgb_config['subsample'],
        'reg_alpha':     xgb_config['reg_alpha'],
    }

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=xgb_config['cv_folds'],
        verbose=0,
        n_jobs=xgb_config['n_jobs']
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print(f" Best Parameters Found: {grid_search.best_params_}")
    print(" Training complete.")
    return best_model