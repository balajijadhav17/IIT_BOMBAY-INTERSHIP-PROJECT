# EEG Emotion Recognition System

## Project Description
This project implements a complete machine learning pipeline for analyzing cognitive states using multi-modal physiological data including EEG, eye tracking, GSR, and facial emotion analysis.

## Features
- Multi-modal data integration (EEG, eye tracking, GSR, facial emotions)
- Advanced ML models (HMM, RNN-HMM hybrid, Seq2Seq with attention)
- 5 cognitive state detection (Focused, Distracted, Fatigued, Engaged, Bored)
- Interactive Streamlit dashboard
- Comprehensive analysis and visualization

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
