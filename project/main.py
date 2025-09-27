# main.py
import sys
import os


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from notebooks.preprocessing import preprocess_data
    from notebooks.feature_engineering import engineer_features
    from notebooks.baseline_HMM import train_baseline_hmm
    from notebooks.rnn_hmm_hybrid import train_rnn_hmm
    from notebooks.seq2seq_attention import train_seq2seq
    from notebooks.analysis_trajectories import analyze_trajectories
    print("All modules imported successfully")
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def main():
    print("Starting EEG Emotion Recognition Pipeline...")
    
    
    print("\n1. Preprocessing data...")
    preprocessed_data = preprocess_data()
   
    print("\n2. Engineering features...")
    engineered_data = engineer_features()
    
    
    print("\n3. Training baseline HMM...")
    hmm_model, hidden_states = train_baseline_hmm()
 
    print("\n4. Training RNN-HMM hybrid model...")
    rnn_hmm_model = train_rnn_hmm()
    
    
    print("\n5. Training Seq2Seq with Attention model...")
    seq2seq_model = train_seq2seq()
    
 
    print("\n6. Analyzing trajectories...")
    analysis_results = analyze_trajectories()
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()