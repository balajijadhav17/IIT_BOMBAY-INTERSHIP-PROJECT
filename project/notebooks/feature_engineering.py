# notebooks/feature_engineering.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pickle

def engineer_features():
  
    with open('../processed/preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    features = data['features']
    
    
    window_size = 10  
    
    eeg_features = features[:, :20]
    eeg_rolling_mean = np.array([np.mean(eeg_features[i-window_size:i, :], axis=0) 
                               for i in range(window_size, len(eeg_features))])
    eeg_rolling_std = np.array([np.std(eeg_features[i-window_size:i, :], axis=0) 
                              for i in range(window_size, len(eeg_features))])
    
    eye_features = features[:, 20:28]
    eye_rolling_mean = np.array([np.mean(eye_features[i-window_size:i, :], axis=0) 
                               for i in range(window_size, len(eye_features))])
    eye_rolling_std = np.array([np.std(eye_features[i-window_size:i, :], axis=0) 
                              for i in range(window_size, len(eye_features))])
    
    
    engineered_features = np.hstack([
        features[window_size:, :],  
        eeg_rolling_mean, eeg_rolling_std,
        eye_rolling_mean, eye_rolling_std
    ])
    
    timestamps = data['timestamps'][window_size:]
    question_keys = data['question_keys'][window_size:]
    
    
    engineered_data = {
        'features': engineered_features,
        'timestamps': timestamps,
        'question_keys': question_keys,
        'scalers': data['scalers']
    }
    
    with open('../processed/engineered_features.pkl', 'wb') as f:
        pickle.dump(engineered_data, f)
    
    print("Feature engineering completed. Data saved to processed/engineered_features.pkl")
    return engineered_data

if __name__ == "__main__":
    engineer_features()