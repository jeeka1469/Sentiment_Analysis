import torch
from model import TransformerClassifier
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifier(vocab_size=30522, hidden_dim=128, num_labels=2, max_len=512).to(device)

# Load model weights
model.load_state_dict(torch.load("Sentiment_Analysis.pth"), strict=False)
model.eval()  # Set the model to evaluation mode

# Test with dummy input (e.g., random tensor of the shape [batch_size, seq_len])
dummy_input = torch.randint(0, 30522, (1, 512)).to(device)  # Example batch size of 1, sequence length of 512
dummy_attention_mask = torch.ones(1, 512).to(device)  # Attention mask for padding (all ones here)

# Run the model
with torch.no_grad():
    output = model(dummy_input, dummy_attention_mask)

# Print the output
print(output)

# Apply softmax to get probabilities
probabilities = F.softmax(output, dim=-1)
print(probabilities)
