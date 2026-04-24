import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_prep_data(x_path, y_path, test_size=0.2, random_state=42):

    print(f"\n Loading features from: {x_path}")
    print(f" Loading labels from: {y_path}")
    # 1. Load the processed arrays
    X_full = np.load(x_path)
    y_full = np.load(y_path)
    
    print("\n Splitting data into train and test sets")
    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=test_size, random_state=random_state
    )
    print(" Scaling features ")
    # 3. Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n Data ready: {X_train_scaled.shape[0]} train samples, {X_test_scaled.shape[0]} test samples.")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler