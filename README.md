# Temporal Emotion Transition Modeling for Student Engagement Analysis 
A comprehensive multi-modal cognitive state recognition system that analyzes EEG, eye tracking, GSR, and facial emotion data to identify patterns in cognitive states and emotional responses using advanced machine learning techniques.


## 🌟 Key Features

### 🔬 Multi-modal Data Integration
- *EEG Data*: Brainwave analysis (Delta, Theta, Alpha, Beta, Gamma bands)
- *Eye Tracking*: Gaze patterns, pupil dilation, fixation metrics
- *GSR (Galvanic Skin Response)*: Emotional arousal measurement
- *Facial Emotion Analysis*: Valence, attention, engagement metrics
- *I-VT Algorithm*: Fixation detection and saccade analysis

### 🤖 Advanced Machine Learning Pipeline
- *Hidden Markov Models (HMM)*: 5-state cognitive modeling
- *RNN-HMM Hybrid*: Temporal sequence modeling with LSTM networks
- *Seq2Seq with Attention*: Transformer-based sequence prediction
- *Real-time State Classification*: Cognitive state identification

### 📈 Interactive Visualization
- *Streamlit Dashboard*: Web-based interactive interface
- *t-SNE Trajectory Plots*: 2D visualization of cognitive state evolution
- *Transition Matrix Analysis*: State transition probabilities
- *Real-time Monitoring*: Live cognitive state tracking

### 🎯 Cognitive State Detection
Identify and classify 5 primary cognitive states with physiological indicators:

| State | Emoji | Description | Key Physiological Indicators |
|-------|--------|-------------|-----------------------------|
| *State 0* | 🎯 | *Focused* | High beta waves, stable gaze, reduced blink rate |
| *State 1* | 😕 | *Distracted* | Increased theta waves, gaze variability, high saccade rate |
| *State 2* | 😴 | *Fatigued* | Elevated alpha waves, slow pupil response, increased blink duration |
| *State 3* | 🤔 | *Engaged* | Moderate beta waves, positive valence, steady attention |
| *State 4* | 🥱 | *Bored* | Reduced physiological responses, gaze disengagement |


## Installation
```bash
pip install -r requirements.txt

## Run full pipeline
cd project python main.py
# After completely run of main.py file then start app.py
streamlit run app.py

## Clone the repository
git clone https://github.com/yourusername/eeg-emotion-recognition.git

# Navigate to project directory
cd eeg-emotion-recognition
