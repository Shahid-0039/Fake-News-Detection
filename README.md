# Fake News Detection

A RoBERTa-based fake news classifier with notebooks and a small Flask demo for submitting article text and receiving a prediction. The project is intended for data scientists learning text classification and for small teams prototyping an ML demo.

## Highlights
- Demonstrates data preprocessing, model training (see `Roberta_base_training.ipynb`), and evaluation.
- Small Flask web app (`Fake_news_Detection/App.py`) that loads a trained model from `Fake_news_Detection/roberta_base/` and serves a prediction UI.

## Stack
- **Languages:** Jupyter Notebook, Python
- **Framework/runtime:** Flask, PyTorch, Hugging Face Transformers
- **Notable libraries:** transformers, torch, scikit-learn, pandas

## Repository layout

```
Fake_news_Detection/            Flask app, notebooks, and web assets
  App.py                        Flask app entrypoint (loads model from roberta_base/)
  requirements.txt              Python dependencies for the project/demo
  Roberta_base_training.ipynb    Notebook showing model training / evaluation
  roberta_base/                  Local model weights / tokenizer (expected)
  templates/                     Flask HTML templates (Index.html, prediction.html)
  static/                        Static assets (images, CSS)
.idea/                           IDE config (can be ignored)
README.md                        Project description and usage (this file)
```

How it fits together: the notebooks explore data and train a RoBERTa-based model. The trained model (in `roberta_base/`) is loaded by `Fake_news_Detection/App.py` and served via Flask; the UI templates in `Fake_news_Detection/templates/` present a compact web interface for users to paste article text and get a classification.

## Quick start — run the demo locally

1. Clone the repository

```bash
git clone https://github.com/Shahid-0039/Fake-News-Detection.git
cd Fake-News-Detection/Fake_news_Detection
```

2. Create a virtual environment and install requirements

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

3. Ensure a trained model is available

The Flask app expects a model directory named `roberta_base/` containing a tokenizer and model files in Hugging Face format. Options:
- Place your model files in `Fake_news_Detection/roberta_base/`.
- Or modify `App.py` to point to a Hugging Face model repo (for example: `"cardiffnlp/twitter-roberta-base"`).

4. Run the demo

```bash
python App.py
# Open http://127.0.0.1:5000/ in a browser
```

Notes
- `App.py` enforces a minimum input length of 20 words before running a prediction; adjust `predict_news` if desired.
- Notebooks can be opened in Jupyter or Jupyter Lab for reproducible experiments.

## Documentation and suggested improvements
Current issues and recommended fixes:
- No LICENSE file: add an explicit license (MIT/Apache-2.0) to clarify reuse.
- Root README was brief and left implied dependencies; I updated it to include clear quick-start steps.
- `Fake_news_Detection/requirements.txt` appears to contain encoding artifacts; please ensure it is valid UTF-8 and optionally move a cleaned `requirements.txt` to the repository root.
- Consider adding CONTRIBUTING.md and a short example input/output in README to help reviewers.
- Large model files should be managed with Git LFS if they remain in the repo.

## Development recommendations
- Add unit tests for `strong_clean` and `predict_news`.
- Provide a small example input (10–20 lines) and expected model output in `examples/`.
- Consider packaging the app with a simple `Dockerfile` for reproducible demos.

## Author
Maintained by Shahid-0039 — https://github.com/Shahid-0039
