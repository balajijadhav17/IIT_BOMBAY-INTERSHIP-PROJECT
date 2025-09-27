# notebooks/rnn_hmm_hybrid.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from config import HMM_PARAMS, RNN_HMM_PARAMS
from utils import save_model

class RNNHMM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.3):
        super(RNNHMM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, 
                          batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.rnn(x, (h0, c0))
        out = self.fc(out)
        out = self.softmax(out)
        return out

def train_rnn_hmm():
   
    with open('../processed/engineered_features.pkl', 'rb') as f:
        data = pickle.load(f)
    
    features = data['features']
    
  
    with open('../processed/hmm_states.pkl', 'rb') as f:
        hmm_data = pickle.load(f)
    
    targets = hmm_data['hidden_states']
    
    
    X = torch.FloatTensor(features).unsqueeze(0)  
    y = torch.LongTensor(targets).unsqueeze(0)

    input_dim = features.shape[1]
    output_dim = HMM_PARAMS['n_components']
    model = RNNHMM(input_dim, RNN_HMM_PARAMS['hidden_dim'], 
                  RNN_HMM_PARAMS['num_layers'], output_dim, 
                  RNN_HMM_PARAMS['dropout'])
    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X)
        loss = criterion(outputs.squeeze(0), y.squeeze(0))
        
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    save_model(model, '../models/rnn_hmm_model.pt')
    print("RNN-HMM training completed. Model saved to models/rnn_hmm_model.pt")
    
    return model

if __name__ == "__main__":
    train_rnn_hmm()