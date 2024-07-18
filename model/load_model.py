from transformers import T5ForConditionalGeneration, T5Tokenizer

def load_model():
    model = T5ForConditionalGeneration.from_pretrained("model/fine_tuned_t5")
    tokenizer = T5Tokenizer.from_pretrained("model/fine_tuned_t5")
    return model, tokenizer
