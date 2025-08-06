import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def load_and_preprocess(data_dir='Data/ProcessedData'):
    # Load California Housing dataset
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    
    # Split into train and test
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # Ensure data_dir exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Save processed data
    X_train_scaled.to_csv(f'{data_dir}/X_train.csv', index=False)
    X_test_scaled.to_csv(f'{data_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{data_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{data_dir}/y_test.csv', index=False)
    
    print("Preprocessing complete. Files saved in", data_dir)

if __name__ == '__main__':
    load_and_preprocess()