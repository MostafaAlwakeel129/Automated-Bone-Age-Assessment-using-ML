from sklearn.ensemble import RandomForestRegressor

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