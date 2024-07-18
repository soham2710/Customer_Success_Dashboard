import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

def load_model():
    model = T5ForConditionalGeneration.from_pretrained("fine_tuned_t5")
    tokenizer = T5Tokenizer.from_pretrained("fine_tuned_t5")
    return model, tokenizer

def preprocess_input(usage_frequency, support_tickets, feedback_score, purchase_amount, tenure):
    return f"{usage_frequency} {support_tickets} {feedback_score} {purchase_amount} {tenure}"
