from fastapi import FastAPI
from pydantic import BaseModel
from src.predict_model import NationalityPredictor

app = FastAPI()

# Inicializa el predictor
predictor = NationalityPredictor("models/nationality_vectorizer.pkl", "models/naive_bayes.pkl")

class Texts(BaseModel):
    texts: list[str]

@app.post("/predict")
def predict_nationality(data: Texts):
    predictions = predictor.predict(data.texts)
    # Asegurarse de que las predicciones sean una lista de strings
    predictions_list = predictions.tolist() if hasattr(predictions, 'tolist') else predictions
    return {"predictions": predictions_list}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
