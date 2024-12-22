import streamlit as st
import torch
from transformers import BertTokenizer
from model import TransformerClassifier  # Import the model class you defined
import torch.nn.functional as F

# Load the model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifier(vocab_size=30522, hidden_dim=128, num_labels=2, max_len=512).to(device)
model.load_state_dict(torch.load("Sentiment_Analysis.pth"), strict=False)
model.eval()  # Set the model to evaluation mode

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Streamlit app UI
st.title("Sentiment Analysis")
st.write("Enter a text for sentiment analysis:")

# Input field for user text
user_input = st.text_area("Text", "I love this! It is amazing.")

if st.button("Analyze"):
    # Preprocess and tokenize the input text
    inputs = tokenizer(user_input, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)

    with torch.no_grad():
        # Forward pass through the model
        outputs = model(inputs['input_ids'], inputs['attention_mask'])

    # Apply softmax to get probabilities
    probabilities = F.softmax(outputs, dim=-1)

    # Get the predicted sentiment based on the highest probability
    predicted_class = torch.argmax(probabilities, dim=-1).item()

    if predicted_class == 0:
        sentiment = "Negative"
    else:
        sentiment = "Positive"

    # Display sentiment and probabilities
    st.write(f"Sentiment: {sentiment}")