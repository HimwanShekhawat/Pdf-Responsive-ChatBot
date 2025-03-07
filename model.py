from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_path = r"G:\Chatbot-using-Lamini-T5-61M-main\LaMini-T5-61M"

# Load locally to verify
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

print("Model loaded successfully!")