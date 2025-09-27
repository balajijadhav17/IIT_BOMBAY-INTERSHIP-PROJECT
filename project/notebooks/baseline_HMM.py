# notebooks/baseline_HMM.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from hmmlearn import hmm
import pickle
from config import HMM_PARAMS
from utils import save_model

def train_baseline_hmm():
   
    with open('../processed/engineered_features.pkl', 'rb') as f:
        data = pickle.load(f)
    
    features = data['features']
   
    model = hmm.GaussianHMM(
        n_components=HMM_PARAMS['n_components'],
        covariance_type=HMM_PARAMS['covariance_type'],
        n_iter=HMM_PARAMS['n_iter']
    )

    model.fit(features)
    

    save_model(model, '../models/hmm_model.pkl')
    
    
    hidden_states = model.predict(features)
    

    with open('../processed/hmm_states.pkl', 'wb') as f:
        pickle.dump({
            'hidden_states': hidden_states,
            'timestamps': data['timestamps'],
            'question_keys': data['question_keys']
        }, f)
    
    print("HMM training completed. Model saved to models/hmm_model.pkl")
    return model, hidden_states

if __name__ == "__main__":
    train_baseline_hmm()