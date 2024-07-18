import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the fine-tuned model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("fine_tuned_t5")
tokenizer = T5Tokenizer.from_pretrained("fine_tuned_t5")

# Streamlit UI
st.title("Customer Success Playbooks Using Predictive Analytics")

st.sidebar.subheader("Customer Features")
usage_frequency = st.sidebar.selectbox("Usage Frequency", ["Daily", "Weekly", "Monthly"])
support_tickets = st.sidebar.slider("Support Tickets", 0, 10, 0)
feedback_score = st.sidebar.slider("Feedback Score", 1.0, 5.0, 3.0, step=0.1)
purchase_amount = st.sidebar.slider("Purchase Amount", 100, 1000, 500, step=10)
tenure = st.sidebar.slider("Tenure (months)", 1, 60, 30)

input_text = f"{usage_frequency} {support_tickets} {feedback_score} {purchase_amount} {tenure}"

# Generate response using T5 model
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
outputs = model.generate(input_ids, max_length=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Display the response
action, email_template = response.split(' ', 1)
st.write(f"Predicted Action: {action}")
st.write(f"Suggested Email Template: {email_template}")

