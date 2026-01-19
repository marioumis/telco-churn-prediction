import pandas as pd
import joblib
import os
# ============================
# 1. Charger les données
# ============================
# ⚠ change le nom du fichier si besoin
df = pd.read_csv(r"../data/telco.csv")

print("Aperçu des données :")
print(df.head())
print("\nInfos générales :")
print(df.info())
print("\nValeurs possibles de Churn :")
print(df["Churn"].value_counts())


# ============================
# 2. Nettoyage minimum
# ============================
# À quoi ça sert ?
# - 'TotalCharges' est parfois lue comme texte, on la convertit en nombre
# - les lignes où ça donne NaN sont supprimées

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

# 'customerID' ne sert pas pour prédire, on l'enlève
if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])


# ============================
# 3. Séparer features (X) et cible (y)
# ============================
# À quoi ça sert ?
# - X = les colonnes qu'on utilise pour prédire
# - y = ce qu'on veut prédire (Churn : Oui/Non → 1/0)

y = df["Churn"].map({"No": 0, "Yes": 1})  # 0 = reste, 1 = quitte
X = df.drop(columns=["Churn"])

print("\nDimensions de X et y :")
print("X :", X.shape)
print("y :", y.shape)


# ============================
# 4. Préparation : colonnes numériques / catégorielles
# ============================
# À quoi ça sert ?
# - les modèles scikit-learn veulent du numérique
# - on va encoder les colonnes de type 'object' (catégorielles)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

print("\nColonnes numériques :", list(num_cols))
print("Colonnes catégorielles :", list(cat_cols))

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),  # on laisse passer les colonnes numériques
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)


# ============================
# 5. Train / Test split + modèle
# ============================
# À quoi ça sert ?
# - on coupe les données en :
#     - train : pour entraîner le modèle
#     - test : pour évaluer sur des données jamais vues

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,          # garde la proportion de churn identique
    random_state=42
)

model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model),
])

# Entraînement
clf.fit(X_train, y_train)
print("\n✅ Modèle entraîné.")


# ============================
# 6. Évaluation du modèle
# ============================
# À quoi ça sert ?
# - voir si le modèle a un minimum de performance
# - accuracy + rapport de classification + matrice de confusion

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = clf.predict(X_test)

print("\nAccuracy sur le test :", accuracy_score(y_test, y_pred))
print("\nClassification report :")
print(classification_report(y_test, y_pred))

print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred))


# ============================
# 7. Sauvegarder le modèle
# ============================
# On sauvegarde dans le même dossier que train_model.py (src/)
BASE_DIR = os.path.dirname(__file__)           # dossier src
model_path = os.path.join(BASE_DIR, "model.joblib")

joblib.dump(clf, model_path)
print(f"💾 Modèle sauvegardé dans {model_path}")
