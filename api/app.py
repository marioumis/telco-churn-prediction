from flask import Flask, request, jsonify
from flask_cors import CORS   # pour autoriser les appels depuis ton front (localhost)
import joblib
import pandas as pd
import os

# ============================
# Charger le modèle
# ============================
# On remonte d'un dossier (..), puis on va dans src/model.joblib
BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # dossier du projet
MODEL_PATH = os.path.join(BASE_DIR, "src", "model.joblib")

model = joblib.load(MODEL_PATH)

# ============================
# Créer l'app Flask + CORS
# ============================
app = Flask(__name__)
CORS(app)  # autorise les requêtes depuis http://localhost:63342 & co


# ============================
# Route de test simple
# ============================
@app.route("/", methods=["GET"])
def home():
    return "API Churn OK ✅"


# ============================
# Route de prédiction
# ============================
@app.route("/predict", methods=["POST"])
def predict():
    """
    Attend un JSON avec les mêmes noms de colonnes que dans le dataset
    (sauf 'Churn').

    Exemple :
    {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        ...
    }
    """
    data = request.get_json()

    if data is None:
        return jsonify({"error": "No JSON received"}), 400

    # On met les données dans un DataFrame avec une seule ligne
    df = pd.DataFrame([data])

    # Prédiction
    proba = model.predict_proba(df)[0][1]   # proba de churn = classe 1
    pred = int(model.predict(df)[0])        # 0 ou 1

    return jsonify({
        "churn_pred": pred,
        "churn_proba": float(proba)
    })


if __name__ == "__main__":
    # debug=True pour voir les erreurs facilement en dev
    app.run(debug=True)
