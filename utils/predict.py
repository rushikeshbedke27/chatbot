import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load label names
dataset = load_dataset("banking77")
label_names = dataset["train"].features["label"].names
num_labels = len(label_names)

# load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

model.load_state_dict(torch.load("model/intent_model.pth", map_location=device))
model.to(device)

model.eval()
CONFIDENCE_THRESHOLD = 0.6

def predict_intent(text:str):
    encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)

    probs = torch.softmax(outputs.logits, dim=1)
    confidence, pred = torch.max(probs, dim=1)

    if confidence.item() < CONFIDENCE_THRESHOLD:
        return {"intent": "fallback", "confidence":confidence.item()}

    return {"intent": label_names[pred.item()], "confidence":confidence.item()} 