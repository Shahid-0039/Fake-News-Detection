# 📰 Fake News Detection

A RoBERTa-based fake news classifier with training notebooks and a lightweight Flask web app for real-time predictions. Paste in an article's text and get an instant classification of whether it's likely real or fake.

Built for data scientists learning text classification workflows and for small teams who want a working reference implementation to prototype from.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Flask](https://img.shields.io/badge/Flask-web%20app-black)
![PyTorch](https://img.shields.io/badge/PyTorch-model-red)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-RoBERTa-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## Overview

This project fine-tunes a RoBERTa transformer model to classify news article text as **real** or **fake**. It includes:

- A training notebook that walks through data preprocessing, model fine-tuning, and evaluation.
- A minimal Flask application that loads the trained model and exposes a simple web UI for submitting article text and receiving a live prediction.

It's designed to be a clear, end-to-end example of taking an NLP model from a notebook to a working demo — useful as a learning reference or as a starting point for a more production-ready system.

## Features

- ✅ End-to-end pipeline: data preprocessing → training → evaluation → deployment
- ✅ Fine-tuned RoBERTa model for binary text classification
- ✅ Simple, self-contained Flask demo with an HTML front end
- ✅ Minimum input-length validation to reduce low-signal predictions
- ✅ Notebook-driven training that's easy to reproduce or adapt to new datasets

## Tech Stack

| Category | Technologies |
|---|---|
| **Languages** | Python, Jupyter Notebook |
| **Web framework** | Flask |
| **ML / DL** | PyTorch, Hugging Face Transformers |
| **Data & utilities** | pandas, scikit-learn |

## Repository Structure

```
Fake-News-Detection/
├── Fake_news_Detection/
│   ├── App.py                       # Flask app entry point — loads the model and serves predictions
│   ├── requirements.txt             # Python dependencies for the app
│   ├── Roberta_base_training.ipynb  # Notebook: preprocessing, training, and evaluation
│   ├── roberta_base/                # Trained model weights + tokenizer (Hugging Face format)
│   ├── templates/                   # Flask HTML templates (Index.html, prediction.html)
│   └── static/                      # CSS, images, and other static assets
├── LICENSE                          # MIT License
└── README.md
```

**How it fits together:** the notebook is used to prepare data and train the RoBERTa-based classifier. The resulting model artifacts are saved into `roberta_base/`, which `App.py` loads at startup. Flask then serves a small web interface (from `templates/`) where a user can paste in article text and receive a prediction.

## How It Works

1. **Preprocessing & training** (`Roberta_base_training.ipynb`) — cleans and tokenizes the dataset, fine-tunes a RoBERTa model for binary classification, and evaluates performance.
2. **Model artifacts** are saved to `Fake_news_Detection/roberta_base/` in standard Hugging Face format (tokenizer + model weights).
3. **Serving** (`App.py`) loads the model at startup and exposes a route that accepts article text, runs it through the model, and returns a real/fake prediction.
4. **UI** (`templates/`, `static/`) renders a simple form for submitting text and displaying the result.

## Getting Started

### Prerequisites

- Python 3.8+
- `pip` and `venv` (or your preferred environment manager)
- A trained model available locally, or a Hugging Face model ID to load instead

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Shahid-0039/Fake-News-Detection.git
cd Fake-News-Detection/Fake_news_Detection

# 2. Create and activate a virtual environment
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

### Model setup

`App.py` expects a model directory at `Fake_news_Detection/roberta_base/` containing tokenizer and model files in Hugging Face format. You have two options:

- **Use a local model:** place your trained tokenizer/model files in `Fake_news_Detection/roberta_base/`.
- **Use a hosted model:** update the model path in `App.py` to point to a Hugging Face Hub model ID instead (e.g. `"cardiffnlp/twitter-roberta-base"`).

### Run the app

```bash
python App.py
```

Then open **http://127.0.0.1:5000/** in your browser.

> **Note:** the app requires a minimum input length (20 words) before running a prediction. You can adjust this in the `predict_news` function in `App.py` if needed.

## Usage

1. Start the Flask app as described above.
2. Paste the text of a news article (20+ words) into the input field.
3. Submit the form to receive a real/fake classification.

For model development, open `Roberta_base_training.ipynb` in Jupyter or JupyterLab to reproduce training or experiment with a different dataset.

## Roadmap

Planned or suggested improvements for contributors:

- [ ] Add unit tests for core functions (`strong_clean`, `predict_news`)
- [ ] Include a small set of example inputs/outputs (e.g. in an `examples/` folder) to help reviewers sanity-check the model
- [ ] Add a `Dockerfile` for reproducible, one-command demos
- [ ] Manage model weights with Git LFS if they're kept in-repo
- [ ] Add a `CONTRIBUTING.md` with setup and contribution guidelines

## Contributing

Contributions, issues, and feature requests are welcome. If you'd like to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Open a pull request describing what you changed and why

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

## Author

Maintained by **[Shahid-0039](https://github.com/Shahid-0039)**
