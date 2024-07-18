import pandas as pd
import numpy as np
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.model_selection import train_test_split
import streamlit as st

# Load and preprocess the dataset
data = {
    'Usage Frequency': ['Daily', 'Weekly', 'Monthly', 'Daily', 'Weekly'],
    'Support Tickets': [6, 1, 4, 2, 6],
    'Feedback Score': [2.5, 4.5, 3.5, 4.5, 2.0],
    'Purchase Amount': [600, 750, 250, 290, 450],
    'Tenure': [15, 25, 5, 13, 10],
    'Action': [
        'Offer intensive support',
        'Offer loyalty rewards',
        'Schedule a feedback session',
        'Send a thank you note',
        'Provide additional resources'
    ],
    'Email Template': [
        "Dear [Customer], We noticed you have been using our product daily and have raised multiple support tickets. To help you better, we are offering an intensive support session tailored to your needs. Please schedule a time that works best for you.",
        "Dear [Customer], Thank you for your continued trust in our product. As a token of our appreciation, we would like to offer you exclusive loyalty rewards. Please check your account for more details.",
        "Dear [Customer], We value your feedback and would like to invite you to a feedback session to discuss your experience with our product. Your insights are crucial for us to improve and serve you better.",
        "Dear [Customer], Thank you for your consistent use of our product. We are thrilled to have you as a customer and appreciate your positive feedback. Please let us know if there is anything more we can do for you.",
        "Dear [Customer], We noticed you have been encountering some issues. To assist you further, we are providing additional resources and guides that might help you navigate through any challenges. Please find the resources attached."
    ]
}

df = pd.DataFrame(data)

# Preprocess data for T5
df['input_text'] = df[['Usage Frequency', 'Support Tickets', 'Feedback Score', 'Purchase Amount', 'Tenure']].astype(str).agg(' '.join, axis=1)
df['target_text'] = df[['Action', 'Email Template']].astype(str).agg(' ', axis=1)

train_df, test_df = train_test_split(df, test_size=0.2)

# Load T5 model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Fine-tuning function
def fine_tune_t5(train_df):
    input_texts = train_df['input_text'].tolist()
    target_texts = train_df['target_text'].tolist()
    
    inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")
    targets = tokenizer(target_texts, padding=True, truncation=True, return_tensors="pt")

    dataset = torch.utils.data.TensorDataset(inputs.input_ids, inputs.attention_mask, targets.input_ids)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(3):
        for batch in loader:
            optimizer.zero_grad()
            input_ids, attention_mask, target_ids = batch
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")

# Fine-tune the model
fine_tune_t5(train_df)

# Save the fine-tuned model
model.save_pretrained("fine_tuned_t5")
tokenizer.save_pretrained("fine_tuned_t5")
