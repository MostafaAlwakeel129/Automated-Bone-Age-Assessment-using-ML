import joblib
import os
from sklearn.metrics import mean_absolute_error, r2_score

def evaluate_and_save(model, X_test, y_test, scaler, exp_dir, model_name="model.pkl", best_params=None):

    print("\nEvaluating model:")
    
    # 1. Make predictions
    preds = model.predict(X_test)
    
    # 2. Calculate metrics
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"Mean Absolute Error: {mae:.2f} months")
    print(f"R-squared Score:     {r2:.4f}")
    
    print("\nSaving model and scaler: ")
    
    # 3. Save scaler
    scaler_path = os.path.join(exp_dir, "feature_scaler.pkl")
    joblib.dump(scaler, scaler_path)

    # 4. Save model — .json for XGBoost, .pkl for everything else
    model_path = os.path.join(exp_dir, model_name)
    if model_name.endswith(".json"):
        model.save_model(model_path)
    else:
        joblib.dump(model, model_path)

    # 5. Save metrics and best params
    metrics_path = os.path.join(exp_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Mean Absolute Error: {mae:.2f} months\n")
        f.write(f"R-squared Score:     {r2:.4f}\n")

        if best_params:
            f.write(f"\nBest Parameters from Grid Search:\n")
            for key, value in best_params.items():
                f.write(f"  {key}: {value}\n")

    print(f"Model, scaler, and metrics saved in: {exp_dir}")