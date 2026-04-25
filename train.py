from boneage_utils_helper import load_yaml, create_experiment_dir
from boneage_utils_data import load_and_prep_data
from boneage_utils_eval import evaluate_and_save
from boneage_utils_models import train_random_forest , train_xgboost

def main():
    print("\n Starting pipeline ")
    
    # 1. Load both Configurations
    config = load_yaml("./config/config.yaml")
    model_configs = load_yaml("./config/model_config.yaml") 
    
    # 2. Create Tracking Folder
    exp_dir = create_experiment_dir(
        base_dir=config['save_dir'], 
        config_dir="./config"
    )
    
    # 3. Load & Scale Data
    X_train, X_test, y_train, y_test, scaler = load_and_prep_data(
        x_path=config['data_paths']['X_data'], 
        y_path=config['data_paths']['y_data'],
        test_size=config['split_params']['test_size'],
        random_state=config['split_params']['random_state']
    )
    
    # 4. Train the model using the YAML parameters
    xgb_model = train_xgboost(X_train, y_train, xgb_config=model_configs['xgboost'])
    evaluate_and_save(xgb_model, X_test, y_test, scaler, exp_dir, model_name="best_boneage_xgb.json")
    
    print("\nPipeline complete, Results saved in:", exp_dir)

if __name__ == "__main__":
    main()