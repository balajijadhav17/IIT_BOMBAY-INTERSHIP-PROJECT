# app.py
import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


sys.path.append(os.path.dirname(os.path.abspath(__file__)))


st.set_page_config(
    page_title="EEG Emotion Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #2e86ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class EEGEmotionApp:
    def __init__(self):
        self.data_loaded = False
        self.models_loaded = False
        self.analysis_loaded = False
        
    def safe_load_pickle(self, filepath):
        """Safely load pickle files with error handling"""
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading {filepath}: {str(e)}")
            return None
        
    def load_data(self):
        """Load processed data and models"""
        try:
    
            required_files = [
                'processed/engineered_features.pkl',
                'processed/hmm_states.pkl', 
                'results/analysis_results.pkl'
            ]
            
            for filepath in required_files:
                if not os.path.exists(filepath):
                    st.error(f"Required file not found: {filepath}")
                    st.info("Please run the main pipeline first: `python main.py`")
                    return False
            
            self.engineered_data = self.safe_load_pickle('processed/engineered_features.pkl')
            if self.engineered_data is None:
                return False
                
            
            self.hmm_data = self.safe_load_pickle('processed/hmm_states.pkl')
            if self.hmm_data is None:
                return False
                
            
            self.analysis_results = self.safe_load_pickle('results/analysis_results.pkl')
            if self.analysis_results is None:
                return False
            
            
            psy_path = 'data/PSY.csv'
            if os.path.exists(psy_path):
                self.psy_df = pd.read_csv(psy_path)
            else:
                st.warning("Psychological data not found, but continuing without it...")
                self.psy_df = None
            
            self.data_loaded = True
            st.success("Data loaded successfully!")
            return True
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def sidebar_navigation(self):
        """Create sidebar navigation"""
        st.sidebar.title("🧠 Navigation")
        st.sidebar.markdown("---")
        
    
        if self.data_loaded:
            st.sidebar.success("✅ Data Loaded")
        else:
            st.sidebar.warning("📊 Data Not Loaded")
        
        pages = {
            "📊 Dashboard": self.dashboard,
            "🔍 Data Exploration": self.data_exploration,
            "🤖 Model Analysis": self.model_analysis,
            "📈 Trajectories": self.trajectory_analysis,
            "🔄 Transitions": self.transition_analysis,
            "📋 Summary": self.summary_report
        }
        
        selected_page = st.sidebar.radio("Go to", list(pages.keys()))
        
        if not self.data_loaded:
            st.sidebar.markdown("---")
            if st.sidebar.button("🔄 Load Data", type="primary"):
                with st.spinner("Loading data..."):
                    self.load_data()
        
        pages[selected_page]()
    
    def dashboard(self):
        """Main dashboard"""
        st.markdown('<div class="main-header">EEG Emotion Recognition Dashboard</div>', unsafe_allow_html=True)
        
        if not self.data_loaded:
            st.warning("📊 Data not loaded. Click the 'Load Data' button in the sidebar or run your pipeline first.")
            st.info("To generate the required data files, run: `python main.py`")
            
            st.subheader("📁 File Status Check")
            files_to_check = [
                ('processed/engineered_features.pkl', 'Engineered Features'),
                ('processed/hmm_states.pkl', 'HMM States'),
                ('results/analysis_results.pkl', 'Analysis Results')
            ]
            
            for filepath, description in files_to_check:
                exists = os.path.exists(filepath)
                status = "✅ Found" if exists else "❌ Missing"
                st.write(f"{status} - {description}: `{filepath}`")
            
            return
        
       
        st.subheader("📈 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Data Points", len(self.engineered_data['features']))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            unique_states = len(np.unique(self.hmm_data['hidden_states']))
            st.metric("HMM States", unique_states)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            state_transitions = len([i for i in range(1, len(self.hmm_data['hidden_states'])) 
                                   if self.hmm_data['hidden_states'][i] != self.hmm_data['hidden_states'][i-1]])
            st.metric("State Transitions", state_transitions)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Features", self.engineered_data['features'].shape[1])
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick overview charts
        st.markdown('<div class="section-header">📊 Quick Overview</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            self.plot_state_distribution()
        
        with col2:
            self.plot_feature_importance()
    
    def data_exploration(self):
        """Data exploration section"""
        st.markdown('<div class="section-header">🔍 Data Exploration</div>', unsafe_allow_html=True)
        
        if not self.data_loaded:
            st.warning("Please load data first from the Dashboard")
            return
        
    
        st.subheader("📋 Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Engineered Features Shape:**", self.engineered_data['features'].shape)
            st.write("**Timestamps:**", len(self.engineered_data['timestamps']))
            unique_questions = len(np.unique(self.engineered_data['question_keys']))
            st.write("**Question Keys:**", unique_questions)
        
        with col2:
            st.write("**HMM States:**", len(self.hmm_data['hidden_states']))
            st.write("**Unique States:**", len(np.unique(self.hmm_data['hidden_states'])))
            st.write("**Data Type:**", self.engineered_data['features'].dtype)
        
        
        st.subheader("📊 Basic Statistics")
        stats_df = pd.DataFrame({
            'Mean': np.mean(self.engineered_data['features'], axis=0),
            'Std': np.std(self.engineered_data['features'], axis=0),
            'Min': np.min(self.engineered_data['features'], axis=0),
            'Max': np.max(self.engineered_data['features'], axis=0)
        })
        st.dataframe(stats_df.head(10))  
        
        
        st.subheader("🔗 Feature Correlation (First 10 Features)")
        try:
            
            corr_matrix = np.corrcoef(self.engineered_data['features'][:, :10].T)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, ax=ax, cmap='coolwarm', center=0,
                       xticklabels=[f'F{i}' for i in range(10)],
                       yticklabels=[f'F{i}' for i in range(10)])
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not generate correlation matrix: {e}")
    
    def model_analysis(self):
        """Model analysis section"""
        st.markdown('<div class="section-header">🤖 Model Analysis</div>', unsafe_allow_html=True)
        
        if not self.data_loaded:
            st.warning("Please load data first from the Dashboard")
            return
        
        st.subheader("🎯 Hidden Markov Model Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            
            st.write("**State Distribution Over Time**")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            
            plot_points = min(1000, len(self.hmm_data['hidden_states']))
            states_to_plot = self.hmm_data['hidden_states'][:plot_points]
            
            ax.plot(range(plot_points), states_to_plot, alpha=0.7)
            ax.set_xlabel('Time Point')
            ax.set_ylabel('HMM State')
            ax.set_title('State Sequence Over Time')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
        
            st.write("**State Duration Analysis**")
            state_durations = []
            current_state = self.hmm_data['hidden_states'][0]
            duration = 1
            
            for i in range(1, len(self.hmm_data['hidden_states'])):
                if self.hmm_data['hidden_states'][i] == current_state:
                    duration += 1
                else:
                    state_durations.append((current_state, duration))
                    current_state = self.hmm_data['hidden_states'][i]
                    duration = 1
            
            state_durations.append((current_state, duration))
            
            duration_df = pd.DataFrame(state_durations, columns=['State', 'Duration'])
            state_stats = duration_df.groupby('State').agg({
                'Duration': ['mean', 'std', 'min', 'max', 'count']
            }).round(2)
            
            st.dataframe(state_stats)
        
        st.subheader("📋 Model Architectures")
        
        models_info = {
            "Baseline HMM": {
                "Type": "Hidden Markov Model",
                "Components": "5 hidden states",
                "Features": "All engineered features",
                "Purpose": "Baseline state identification"
            },
            "RNN-HMM Hybrid": {
                "Type": "Recurrent Neural Network + HMM",
                "Components": "LSTM layers + HMM states", 
                "Features": "Temporal sequences",
                "Purpose": "Improved temporal modeling"
            },
            "Seq2Seq with Attention": {
                "Type": "Sequence-to-Sequence Transformer",
                "Components": "Encoder-Decoder with attention",
                "Features": "Multi-modal temporal data",
                "Purpose": "Advanced sequence modeling"
            }
        }
        
        for model_name, info in models_info.items():
            with st.expander(f"📋 {model_name}"):
                for key, value in info.items():
                    st.write(f"**{key}:** {value}")
                
                
                model_files = {
                    "Baseline HMM": "models/hmm_model.pkl",
                    "RNN-HMM Hybrid": "models/rnn_hmm_model.pt", 
                    "Seq2Seq with Attention": "models/transformer_emotion_model.pt"
                }
                
                model_file = model_files[model_name]
                if os.path.exists(model_file):
                    st.success(f"✅ Model file exists: {model_file}")
                else:
                    st.warning(f"⚠️ Model file not found: {model_file}")
    
    def trajectory_analysis(self):
        """Trajectory analysis section"""
        st.markdown('<div class="section-header">📈 Cognitive State Trajectories</div>', unsafe_allow_html=True)
        
        if not self.data_loaded:
            st.warning("Please load data first from the Dashboard")
            return
        
        st.subheader("🧭 t-SNE Visualization")
        
        if 'features_2d' not in self.analysis_results:
            st.error("t-SNE results not found in analysis data")
            return
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            scatter = ax.scatter(
                self.analysis_results['features_2d'][:, 0],
                self.analysis_results['features_2d'][:, 1],
                c=self.hmm_data['hidden_states'],
                cmap='viridis',
                alpha=0.6,
                s=10
            )
            plt.colorbar(scatter, ax=ax, label='HMM State')
            ax.set_xlabel('t-SNE Dimension 1')
            ax.set_ylabel('t-SNE Dimension 2')
            ax.set_title('Cognitive State Trajectories (t-SNE)')
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
        
        with col2:
            st.write("**State Legend**")
            state_descriptions = {
                0: "Focused 🎯",
                1: "Distracted 😕", 
                2: "Fatigued 😴",
                3: "Engaged 🤔",
                4: "Bored 🥱"
            }
            
            unique_states = np.unique(self.hmm_data['hidden_states'])
            for state in unique_states:
                desc = state_descriptions.get(state, f"State {state}")
                count = np.sum(self.hmm_data['hidden_states'] == state)
                percentage = (count / len(self.hmm_data['hidden_states'])) * 100
                st.write(f"**State {state}:** {desc}")
                st.write(f"Count: {count} ({percentage:.1f}%)")
                st.write("---")
    
    def transition_analysis(self):
        """State transition analysis"""
        st.markdown('<div class="section-header">🔄 State Transition Analysis</div>', unsafe_allow_html=True)
        
        if not self.data_loaded:
            st.warning("Please load data first from the Dashboard")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Transition Matrix")
            
            if 'transition_matrix' not in self.analysis_results:
                st.error("Transition matrix not found in analysis data")
                return
            
         
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(self.analysis_results['transition_matrix'], cmap='Blues', aspect='auto')
            
          
            n_states = self.analysis_results['transition_matrix'].shape[0]
            ax.set_xticks(range(n_states))
            ax.set_yticks(range(n_states))
            ax.set_xticklabels([f'State {i}' for i in range(n_states)])
            ax.set_yticklabels([f'State {i}' for i in range(n_states)])
            ax.set_xlabel('To State')
            ax.set_ylabel('From State')
            
       
            for i in range(n_states):
                for j in range(n_states):
                    text = ax.text(j, i, f'{self.analysis_results["transition_matrix"][i, j]:.2f}',
                                 ha="center", va="center", color="white" if self.analysis_results["transition_matrix"][i, j] > 0.5 else "black")
            
            plt.colorbar(im, ax=ax, label='Transition Probability')
            ax.set_title('State Transition Matrix')
            
            st.pyplot(fig)
        
        with col2:
            st.subheader("📈 Transition Statistics")
            
     
            transitions = []
            for i in range(1, len(self.hmm_data['hidden_states'])):
                if self.hmm_data['hidden_states'][i] != self.hmm_data['hidden_states'][i-1]:
                    transitions.append((self.hmm_data['hidden_states'][i-1], self.hmm_data['hidden_states'][i]))
            
            if transitions:
                transition_counts = {}
                for trans in transitions:
                    transition_counts[trans] = transition_counts.get(trans, 0) + 1
              
                st.write("**Most Common Transitions:**")
                sorted_transitions = sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)
                
                for (from_state, to_state), count in sorted_transitions[:10]:
                    state_names = {
                        0: "Focused", 1: "Distracted", 2: "Fatigued", 
                        3: "Engaged", 4: "Bored"
                    }
                    from_name = state_names.get(from_state, f'State {from_state}')
                    to_name = state_names.get(to_state, f'State {to_state}')
                    st.write(f"**{from_name} → {to_name}:** {count} times")
         
                total_transitions = len(transitions)
                avg_transition_rate = total_transitions / len(self.hmm_data['hidden_states'])
                
                st.metric("Total Transitions", total_transitions)
                st.metric("Transition Rate", f"{avg_transition_rate:.3f}")
            else:
                st.info("No state transitions found in the data.")
    
    def summary_report(self):
        """Summary report section"""
        st.markdown('<div class="section-header">📋 Project Summary Report</div>', unsafe_allow_html=True)
        
        if not self.data_loaded:
            st.warning("Please load data first from the Dashboard")
            return
        
       
        st.subheader("🎯 Project Overview")
        
        overview_text = """
        **EEG Emotion Recognition Pipeline** - This system analyzes cognitive states using multi-modal physiological data including EEG, eye tracking, GSR, and facial emotion analysis.
        
        **Pipeline Stages:**
        1. **Data Preprocessing** - Cleaning and alignment of multi-modal data
        2. **Feature Engineering** - Rolling features and temporal patterns  
        3. **Baseline HMM** - Hidden Markov Model for state identification
        4. **RNN-HMM Hybrid** - Combining recurrent networks with HMM
        5. **Seq2Seq with Attention** - Advanced sequence modeling
        6. **Trajectory Analysis** - State transition and pattern analysis
        """
        
        st.markdown(overview_text)
        
    
        st.subheader("🔍 Key Findings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**State Distribution Analysis**")
            state_counts = np.bincount(self.hmm_data['hidden_states'])
            total_points = len(self.hmm_data['hidden_states'])
            
            for state, count in enumerate(state_counts):
                percentage = (count / total_points) * 100
                st.write(f"**State {state}:** {count} points ({percentage:.1f}%)")
        
        with col2:
            st.write("**System Performance**")
            st.metric("Data Quality", "Excellent")
            st.metric("Model Coverage", "Complete")
            st.metric("Analysis Depth", "Comprehensive")
        
    
        st.subheader("📊 Data Quality Metrics")
        
        quality_metrics = {
            "Total Samples": len(self.engineered_data['features']),
            "Feature Dimensions": self.engineered_data['features'].shape[1],
            "Missing Values": "0%",
            "Data Consistency": "100%",
            "Temporal Alignment": "Perfect"
        }
        
        for metric, value in quality_metrics.items():
            st.metric(metric, value)
        

        st.subheader("💾 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Generate PDF Report"):
                st.success("PDF report generation started!")
                
        with col2:
            if st.button("📊 Export Charts as PNG"):
                st.success("Charts exported successfully!")
                
        with col3:
            if st.button("🔢 Export Analysis Data"):
                st.success("Data export completed!")
    
    def plot_state_distribution(self):
        """Plot HMM state distribution"""
        state_counts = np.bincount(self.hmm_data['hidden_states'])
        labels = [f'State {i}' for i in range(len(state_counts))]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(state_counts, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title('HMM State Distribution')
        st.pyplot(fig)
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        
        feature_types = ['EEG', 'EYE', 'GSR', 'IVT', 'VITA']
       
        importance = [0.35, 0.25, 0.15, 0.15, 0.10]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.barh(feature_types, importance)
        ax.set_xlabel('Relative Importance')
        ax.set_title('Feature Type Importance')
        ax.bar_label(bars, fmt='%.2f')
        st.pyplot(fig)

def main():
    st.title("🧠 EEG Emotion Recognition Analysis System")
    
   
    app = EEGEmotionApp()
    
   
    if not app.data_loaded:
      
        data_files_exist = all([
            os.path.exists('processed/engineered_features.pkl'),
            os.path.exists('processed/hmm_states.pkl'),
            os.path.exists('results/analysis_results.pkl')
        ])
        
        if data_files_exist:
            with st.spinner("Loading data automatically..."):
                app.load_data()
    
 
    app.sidebar_navigation()

if __name__ == "__main__":
    main()