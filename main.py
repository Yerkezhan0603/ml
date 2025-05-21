from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib

app = FastAPI()

app.mount("/static", StaticFiles(directory="frontend"), name="static")
templates = Jinja2Templates(directory="frontend")

model = joblib.load("backend/model.joblib")
vectorizer = joblib.load("backend/tfidf.joblib")
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Store user input in memory (simplified persistence)
last_input = {"text": ""}

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request, clear: bool = False):
    if clear:
        last_input["text"] = ""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": None,
        "user_input": last_input["text"]
    })

@app.post("/", response_class=HTMLResponse)
def form_post(request: Request, comment: str = Form(...)):
    last_input["text"] = comment
    vec = vectorizer.transform([comment])
    preds = model.predict(vec)[0]
    result = [label for label, p in zip(labels, preds) if p == 1]
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": result,
        "user_input": comment
    })
