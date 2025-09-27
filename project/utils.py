# utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import torch
import os

def load_data(file_path):
    """Load data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def normalize_data(data):
    """Normalize data using StandardScaler"""
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data, scaler

def save_model(model, file_path):
    """Save model to file"""

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        if file_path.endswith('.pkl'):
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
        elif file_path.endswith('.pt'):
            torch.save(model.state_dict(), file_path)
        print(f"Model saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving model to {file_path}: {str(e)}")

def load_model(file_path, model_class=None):
    """Load model from file"""
    try:
        if file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        elif file_path.endswith('.pt') and model_class:
            model = model_class()
            model.load_state_dict(torch.load(file_path))
            return model
    except Exception as e:
        print(f"Error loading model from {file_path}: {str(e)}")
        return None

def align_data_by_timestamp(dataframes, timestamp_col='UnixTime'):
    """Align multiple dataframes by timestamp"""
    if not dataframes:
        return []
    

    common_ts = set(dataframes[0][timestamp_col])
    for df in dataframes[1:]:
        common_ts = common_ts.intersection(set(df[timestamp_col]))
    
    if not common_ts:
        print("Warning: No common timestamps found across dataframes")
        return dataframes
    
    common_ts = sorted(common_ts)
    
   
    aligned_dfs = []
    for i, df in enumerate(dataframes):
        aligned_df = df[df[timestamp_col].isin(common_ts)].sort_values(timestamp_col)
        aligned_df = aligned_df.reset_index(drop=True)
        aligned_dfs.append(aligned_df)
        print(f"Dataframe {i+1}: {len(aligned_df)} rows after alignment")
    
    return aligned_dfs

def create_directory(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)
    print(f"Directory created or already exists: {path}")

def check_file_exists(file_path):
    """Check if a file exists"""
    exists = os.path.isfile(file_path)
    if exists:
        print(f"File exists: {file_path}")
    else:
        print(f"File does not exist: {file_path}")
    return exists

def get_data_summary(df, name=""):
    """Get summary of a dataframe"""
    if df is None:
        print(f"No data available for {name}")
        return
    
    print(f"\n=== {name} Data Summary ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Data types:\n{df.dtypes}")
    
   
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\nBasic statistics for numeric columns:")
        print(df[numeric_cols].describe())