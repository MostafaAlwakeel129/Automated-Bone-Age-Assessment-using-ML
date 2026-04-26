import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from boneage_utils_data import load_and_prep_data
from boneage_utils_helper import load_yaml

config = load_yaml("./config/config.yaml")

X_train, X_test, y_train, y_test, scaler = load_and_prep_data(
    x_path=config['data_paths']['X_data'],
    y_path=config['data_paths']['y_data'],
    test_size=config['split_params']['test_size'],
    random_state=config['split_params']['random_state']
)

male_model   = joblib.load("./models/2026_04_26_14_31/tuned_rf_male.pkl")
female_model = joblib.load("./models/2026_04_26_14_31/tuned_rf_female.pkl")

male_mask   = X_test[:, -1] > 0
female_mask = X_test[:, -1] <= 0

all_preds = np.empty(len(y_test))
all_preds[male_mask]   = male_model.predict(X_test[male_mask])
all_preds[female_mask] = female_model.predict(X_test[female_mask])

mae = mean_absolute_error(y_test, all_preds)
r2  = r2_score(y_test, all_preds)
print(f"Combined MAE: {mae:.2f} months")
print(f"Combined R²:  {r2:.4f}")