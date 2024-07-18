import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.model_selection import train_test_split
import pandas as pd

def fine_tune_t5(train_df):
    # Load T5 model and tokenizer
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

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

    # Save the fine-tuned model
    model.save_pretrained("model/fine_tuned_t5")
    tokenizer.save_pretrained("model/fine_tuned_t5")

if __name__ == "__main__":
    # Load your preprocessed dataset here
    additional_data = {
        'Usage Frequency': ['Monthly', 'Weekly', 'Daily', 'Monthly', 'Weekly'],
        'Support Tickets': [1, 5, 2, 0, 6],
        'Feedback Score': [4.0, 2.5, 5.0, 3.5, 2.0],
        'Purchase Amount': [200, 650, 800, 120, 500],
        'Tenure': [6, 20, 30, 3, 15],
        'Action': [
            'Offer a product demo',
            'Provide personalized support',
            'Invite to customer success webinar',
            'Send promotional offers',
            'Schedule a follow-up call'
        ],
        'Email Template': [
            "Dear [Customer], As a valued monthly user, we would like to offer you a product demo to help you get the most out of our product. Please let us know a convenient time for you.",
            "Dear [Customer], We noticed you've been experiencing some issues. Our support team is here to provide personalized assistance. Please contact us at your earliest convenience.",
            "Dear [Customer], We are excited to invite you to our upcoming customer success webinar, where we will share tips and best practices for using our product effectively. Please join us on [date].",
            "Dear [Customer], We have some exciting promotional offers just for you! Check your account for exclusive discounts and offers.",
            "Dear [Customer], We would like to schedule a follow-up call to discuss your experience with our product and address any concerns you may have. Please let us know a suitable time for you."
        ]
    }
    df_additional = pd.DataFrame(additional_data)
    
    # Fine-tune the T5 model
    fine_tune_t5(df_additional)
