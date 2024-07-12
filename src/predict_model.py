import joblib

class NationalityPredictor:
    def __init__(self, vectorizer_path, model_path):
        self.vectorizer = self._load_pickle(vectorizer_path)
        self.model = self._load_pickle(model_path)

    def _load_pickle(self, path):
        with open(path, "rb") as file:
            return joblib.load(file)

    def predict(self, texts):
        transformed_texts = self.vectorizer.transform(texts)
        predictions = self.model.predict(transformed_texts)
        return predictions
