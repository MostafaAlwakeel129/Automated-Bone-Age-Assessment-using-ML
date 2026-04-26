from boneage_utils_helper import load_yaml, create_experiment_dir
from boneage_utils_data import load_and_prep_data
from boneage_utils_eval import evaluate_and_save
from boneage_utils_models import train_gender_split
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def main():
    print("\n Starting enhanced pipeline ")

    # 1. Load both configurations 
    config = load_yaml("./config/config.yaml")
    model_configs = load_yaml("./config/model_config.yaml")

    # 2. Create tracking folder 
    exp_dir = create_experiment_dir(
        base_dir=config['save_dir'],
        config_dir="./config"
    )

    # 3. Load & scale data 
    X_train, X_test, y_train, y_test, scaler = load_and_prep_data(
        x_path=config['data_paths']['X_data'],
        y_path=config['data_paths']['y_data'],
        test_size=config['split_params']['test_size'],
        random_state=config['split_params']['random_state']
    )

    # 4. Train gender-split tuned models (Enhancement 1 + 2 combined)
    (male_model,   X_test_male,   y_test_male,
     female_model, X_test_female, y_test_female) = train_gender_split(
        X_train, y_train, X_test, y_test,
        rf_config=model_configs['random_forest']
    )

    # 5. Evaluate and save each model separately
    print("\n--- Male model results ---")
    evaluate_and_save(male_model,   X_test_male,   y_test_male,
                      scaler, exp_dir, model_name="tuned_rf_male.pkl")

    print("\n--- Female model results ---")
    evaluate_and_save(female_model, X_test_female, y_test_female,
                      scaler, exp_dir, model_name="tuned_rf_female.pkl")

    # 6. Combined score: merge predictions from both models on the full test set
    print("\n--- Combined (overall) results ---")
    all_preds = np.empty(len(y_test))

    # Rebuild the gender masks for the full test set
    male_mask   = X_test[:, -1] == 1.0
    female_mask = X_test[:, -1] == 0.0
    all_preds[male_mask]   = male_model.predict(X_test[male_mask])
    all_preds[female_mask] = female_model.predict(X_test[female_mask])

    combined_mae = mean_absolute_error(y_test, all_preds)
    combined_r2  = r2_score(y_test, all_preds)
    print(f"Final Mean Absolute Error: {combined_mae:.2f} months")
    print(f"Final R-squared Score:     {combined_r2:.4f}")

    # Save combined metrics
    with open(f"{exp_dir}/metrics_combined.txt", "w") as f:
        f.write(f"Final Mean Absolute Error: {combined_mae:.2f} months\n")
        f.write(f"Final R-squared Score: {combined_r2:.4f}\n")

    print(f"\nPipeline complete. Results saved in: {exp_dir}")


if __name__ == "__main__":
    main()