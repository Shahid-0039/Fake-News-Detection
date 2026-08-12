```python
import streamlit as st
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

MODEL_PATH = "roberta_base"


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model


# --------------------------------------------------
# Text Cleaning
# --------------------------------------------------

def strong_clean(text):
    text = text.lower()

    text = re.sub(
        r'\b(facebook|share|click|subscribe|viral|subscribe)\b',
        '',
        text
    )

    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_news(text, tokenizer, model):

    if len(text.split()) < 20:
        return None, "Please enter at least 20 words."

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

        probabilities = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1).item()

        confidence = probabilities[0][pred].item() * 100

    label_map = {
        0: "Fake News",
        1: "Real News"
    }

    result = label_map.get(pred, "Unknown")

    return result, confidence


# --------------------------------------------------
# Load Model
# --------------------------------------------------

try:
    tokenizer, model = load_model()
except Exception as e:
    st.error("Unable to load the model.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# User Interface
# --------------------------------------------------

st.title("📰 Fake News Detection")

st.write(
    "Enter a news article below and the RoBERTa-based "
    "classifier will predict whether it is likely to be "
    "Fake News or Real News."
)

st.info("Please enter at least 20 words for prediction.")


news_text = st.text_area(
    "Enter News Article",
    height=250,
    placeholder="Paste the news article here..."
)


# --------------------------------------------------
# Prediction Button
# --------------------------------------------------

if st.button("🔍 Detect News", use_container_width=True):

    if not news_text.strip():

        st.warning("Please enter a news article.")

    else:

        result, confidence = predict_news(
            news_text,
            tokenizer,
            model
        )

        if result is None:

            st.warning(confidence)

        elif result == "Fake News":

            st.error(f"🚨 {result}")

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        else:

            st.success(f"✅ {result}")

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Powered by RoBERTa • Fake News Detection Project"
)
```
