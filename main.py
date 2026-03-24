from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

app = FastAPI()

# Request body structure
class EmailRequest(BaseModel):
    email_text: str

# Prediction endpoint
@app.post("/predict")
def predict(request: EmailRequest):
    text_vectorized = vectorizer.transform([request.email_text])
    prediction = model.predict(text_vectorized)[0]
    confidence = max(model.predict_proba(text_vectorized)[0]) * 100
    
    return {
        "prediction": "Spam" if prediction == 1 else "Not Spam",
        "confidence": round(confidence, 1),
        "is_spam": bool(prediction)
    }

# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")