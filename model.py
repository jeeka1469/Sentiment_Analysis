# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_labels, max_len):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8), num_layers=6)
        self.fc = nn.Linear(hidden_dim, num_labels)
        self.max_len = max_len
    
    def forward(self, input_ids, attention_mask):
        embeddings = self.embedding(input_ids)
        encoded = self.encoder(embeddings)
        pooled = encoded.mean(dim=1)
        output = self.fc(pooled)
        return output