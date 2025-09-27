# notebooks/preprocessing.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pickle
from config import DATA_PATHS
from utils import load_data, normalize_data, align_data_by_timestamp

def preprocess_data():

    psy_df = load_data(DATA_PATHS['psy'])
    eeg_df = load_data(DATA_PATHS['eeg'])
    gsr_df = load_data(DATA_PATHS['gsr'])
    eye_df = load_data(DATA_PATHS['eye'])
    ivt_df = load_data(DATA_PATHS['ivt'])
    vita_df = load_data(DATA_PATHS['vita'])
    
    
    for df in [eeg_df, gsr_df, eye_df, ivt_df, vita_df]:
        if 'Unnamed: 0' in df.columns:
            df.drop('Unnamed: 0', axis=1, inplace=True)
    
    
    aligned_dfs = align_data_by_timestamp([eeg_df, gsr_df, eye_df, ivt_df, vita_df])
    eeg_df, gsr_df, eye_df, ivt_df, vita_df = aligned_dfs
    
    
    from config import EEG_FEATURES, EYE_FEATURES, GSR_FEATURES, IVT_FEATURES, VITA_FEATURES
    
    eeg_features = eeg_df[EEG_FEATURES]
    eye_features = eye_df[EYE_FEATURES]
    gsr_features = gsr_df[GSR_FEATURES]
    ivt_features = ivt_df[IVT_FEATURES]
    vita_features = vita_df[VITA_FEATURES]
    

    eeg_norm, eeg_scaler = normalize_data(eeg_features)
    eye_norm, eye_scaler = normalize_data(eye_features)
    gsr_norm, gsr_scaler = normalize_data(gsr_features)
    ivt_norm, ivt_scaler = normalize_data(ivt_features)
    vita_norm, vita_scaler = normalize_data(vita_features)
    

    all_features = np.hstack([eeg_norm, eye_norm, gsr_norm, ivt_norm, vita_norm])
    

    processed_data = {
        'features': all_features,
        'timestamps': eeg_df['UnixTime'].values,
        'question_keys': eeg_df['QuestionKey'].values,
        'scalers': {
            'eeg': eeg_scaler,
            'eye': eye_scaler,
            'gsr': gsr_scaler,
            'ivt': ivt_scaler,
            'vita': vita_scaler
        }
    }
    

    os.makedirs('../processed', exist_ok=True)
    with open('../processed/preprocessed_data.pkl', 'wb') as f:
        pickle.dump(processed_data, f)
    
    print("Preprocessing completed. Data saved to processed/preprocessed_data.pkl")
    return processed_data

if __name__ == "__main__":
    preprocess_data()