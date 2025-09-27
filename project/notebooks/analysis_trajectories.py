# notebooks/analysis_trajectories.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pickle

def analyze_trajectories():
    
    with open('../processed/engineered_features.pkl', 'rb') as f:
        data = pickle.load(f)
    
    features = data['features']
    timestamps = data['timestamps']
    question_keys = data['question_keys']
    
    with open('../processed/hmm_states.pkl', 'rb') as f:
        hmm_data = pickle.load(f)
    
    hmm_states = hmm_data['hidden_states']
    
    
    
    psy_df = pd.read_csv('D:/last_second/project/data/PSY.csv')
    
    
    print("Performing dimensionality reduction...")
    
    
    pca = PCA(n_components=50)
    features_pca = pca.fit_transform(features)
    
    
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features_pca)
    

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=hmm_states, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='HMM State')
    plt.title('Cognitive State Trajectories (Colored by HMM State)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')    
    results_dir = 'D:/last_second/project/results'
    if not os.path.exists(results_dir):
         os.makedirs(results_dir)
    plt.savefig(os.path.join(results_dir, 'trajectories_hmm.png'))
    
    plt.close()

    state_transitions = []
    for i in range(1, len(hmm_states)):
        if hmm_states[i] != hmm_states[i-1]:
            state_transitions.append((i, hmm_states[i-1], hmm_states[i]))
    
    n_states = len(np.unique(hmm_states))
    transition_matrix = np.zeros((n_states, n_states))
    
    for i in range(1, len(hmm_states)):
        prev_state = hmm_states[i-1]
        curr_state = hmm_states[i]
        transition_matrix[prev_state, curr_state] += 1
    
    
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = transition_matrix / row_sums

    
    plt.figure(figsize=(10, 8))
    sns.heatmap(transition_matrix, annot=True, cmap='Blues', fmt='.2f')
    plt.title('State Transition Matrix')
    plt.xlabel('Next State')
    plt.ylabel('Current State')
     # plt.savefig('../results/transition_matrix.png')
    plt.savefig('D:/last_second/project/results/transition_matrix.png')
    plt.close()
    
    
    analysis_results = {
        'features_2d': features_2d,
        'hmm_states': hmm_states,
        'transition_matrix': transition_matrix,
        'state_transitions': state_transitions
    }
    
    os.makedirs('../results', exist_ok=True)
    
    with open('../results/analysis_results.pkl', 'wb') as f:
        pickle.dump(analysis_results, f)

    with open('../results/analysis_summary.txt', 'w') as f:
        f.write("Cognitive State Trajectory Analysis Summary\n")
        f.write("===========================================\n\n")
        f.write(f"Total data points: {len(features)}\n")
        f.write(f"Number of HMM states: {n_states}\n")
        f.write(f"Number of state transitions: {len(state_transitions)}\n\n")
        
        f.write("State Transition Probabilities:\n")
        for i in range(n_states):
            for j in range(n_states):
                f.write(f"P(State {j} | State {i}) = {transition_matrix[i, j]:.3f}\n")
            f.write("\n")
        
        f.write("Most Common Transitions:\n")
        transition_counts = {}
        for trans in state_transitions:
            key = (trans[1], trans[2])
            transition_counts[key] = transition_counts.get(key, 0) + 1
        
        sorted_transitions = sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)
        for (from_state, to_state), count in sorted_transitions[:10]:
            f.write(f"From State {from_state} to State {to_state}: {count} times\n")
    
    print("Analysis completed. Results saved to results/ directory.")
    return analysis_results

if __name__ == "__main__":
    analyze_trajectories()