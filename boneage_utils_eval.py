import joblib
import os
from sklearn.metrics import mean_absolute_error, r2_score

def evaluate_and_save(model, X_test, y_test, scaler, exp_dir, model_name="model.pkl"):

    print("\n evaluating model")
    
    # 1. Make predictions
    preds = model.predict(X_test)
    
    # 2. Calculate metrics
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"Mean Absolute Error: {mae:.2f} months")
    print(f"R-squared Score:     {r2:.4f}")
    
    # 3. Save the model and the scaler
    print("\n Saving model and scaler ")
    model_path = os.path.join(exp_dir, model_name)
    scaler_path = os.path.join(exp_dir, "feature_scaler.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"model and scaler saved successfully in: {exp_dir}")