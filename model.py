from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_PATH = "/app/LaMini-T5-61M"

# Load locally to verify
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

print("Model loaded successfully!")