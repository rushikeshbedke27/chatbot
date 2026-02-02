from fastapi import FastAPI
from pydantic import BaseModel
from utils.predict import predict_intent

app = FastAPI(title="Intent Classification API")

class UserInput(BaseModel):
    text: str

@app.post("/predict")
def predict(data: UserInput):
    result = predict_intent(data.text)
    return {"input_text": data.text, "intent": result["intent"], 
            "confidence":round(result["confidence"], 3)}
