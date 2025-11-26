from flask import Flask, render_template, request
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# -------------------------
# Load model
# -------------------------
MODEL_PATH = "roberta_base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()


# -------------------------
# Clean text (same as training)
# -------------------------
def strong_clean(text):
    text = text.lower()
    text = re.sub(r'\b(facebook|share|click|subscribe|viral|subscribe)\b', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# -------------------------
# Predict
# -------------------------
def predict_news(text):
    if len(text.split()) < 20:
        return "Please enter at least 20 words."

    cleaned = strong_clean(text)

    encoded = tokenizer(
        cleaned,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=192
    )

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=1).item()

    label_map = {0: "Fake News", 1: "Real News"}
    return label_map.get(pred, "Unknown")


# -------------------------
# Flask
# -------------------------
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("Index.html")


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    result_text = None

    if request.method == "POST":
        news_text = request.form["news"]
        result_text = predict_news(news_text)

    return render_template("prediction.html", prediction_text=result_text)


if __name__ == "__main__":
    app.run(debug=True)
