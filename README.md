# Email Spam Detector — FastAPI

A production-style ML REST API that detects email spam using Naive Bayes and TF-IDF, served via FastAPI with a clean HTML frontend.

## Live API Documentation
Run locally and visit http://127.0.0.1:8000/docs

## Problem Statement
Streamlit ML apps cannot be integrated into other applications, mobile apps or websites. Real companies need ML models deployed as APIs so any app can call them.

## Solution
The same Naive Bayes spam detection model served as a REST API using FastAPI — making it accessible to any application that can send an HTTP request.

## Features
- REST API endpoint for spam prediction
- Returns prediction, confidence score and is_spam boolean
- Clean HTML frontend to test predictions
- Auto-generated Swagger UI API documentation
- 96.2% accuracy on 5,572 real emails

## Tech Stack
- Python
- FastAPI
- Uvicorn
- Scikit-learn
- Naive Bayes
- TF-IDF Vectorizer
- HTML and JavaScript

## API Endpoint
POST /predict

Request:
```json
{
  "email_text": "Your email text here"
}
```

Response:
```json
{
  "prediction": "Spam",
  "confidence": 93.4,
  "is_spam": true
}
```

## How to Run Locally
```bash
git clone https://github.com/Abhisyanth-M/spam-detector-fastapi
cd spam-detector-fastapi
pip install -r requirements.txt
python train_model.py
python -m uvicorn main:app --reload
```

Then open http://127.0.0.1:8000 in your browser.

## Limitations
- Model requires retraining if spam.csv dataset is updated
- Not deployed to cloud yet — runs locally only

## GitHub
https://github.com/Abhisyanth-M/spam-detector-fastapi
```
