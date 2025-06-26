from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

# Model path
model_path = r"C:\Users\rithvik\OneDrive\Desktop\CCBD CDSAML\models\best_bert_finetuned_model"

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Emotion labels
label_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).squeeze()

        predicted_class = torch.argmax(probs).item()
        confidence = probs[predicted_class].item() * 100  # Convert to %
    
    return label_names[predicted_class], confidence

# Loop for continuous input
print("Enter text to predict emotion (type 'exit' to quit):")
while True:
    user_input = input("> ")
    if user_input.strip().lower() == "exit":
        print("Exiting.")
        break
    emotion, confidence = predict_emotion(user_input)
    print(f"Predicted Emotion: {emotion} ({confidence:.2f}%)")
