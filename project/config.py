DATA_PATHS = {
    'psy': 'data/PSY.csv',
    'eeg': 'data/EEG.csv',
    'gsr': 'data/GSR.csv',
    'eye': 'data/EYE.csv',
    'ivt': 'data/IVT.csv',
    'vita': 'data/VITA.csv'
}


EEG_FEATURES = ['Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10',
               'Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10',
               'Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10',
               'Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10',
               'Gamma_TP9', 'Gamma_AF7', 'Gamma_AF8', 'Gamma_TP10']


EYE_FEATURES = ['ET_GazeLeftx', 'ET_GazeLefty', 'ET_GazeRightx', 'ET_GazeRighty',
               'ET_PupilLeft', 'ET_PupilRight', 'ET_DistanceLeft', 'ET_DistanceRight']


GSR_FEATURES = ['GSR Resistance CAL', 'GSR Conductance CAL']


IVT_FEATURES = ['Gaze X', 'Gaze Y', 'Fixation Duration', 'Saccade Amplitude', 'Gaze Velocity']


VITA_FEATURES = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Joy', 
                'Sadness', 'Surprise', 'Engagement', 'Valence', 'Attention']


HMM_PARAMS = {
    'n_components': 5,
    'covariance_type': 'diag',
    'n_iter': 1000
}

RNN_HMM_PARAMS = {
    'hidden_dim': 64,
    'num_layers': 2,
    'dropout': 0.3
}