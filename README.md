# 📉 Telco Customer Churn Prediction

A machine learning project that predicts customer churn in the telecommunications sector.
The goal is to identify customers who are likely to leave a service based on their usage and contract data.

---

## 🧠 Overview
Customer churn is a major challenge for telecom companies.
This project demonstrates an end-to-end machine learning workflow:
- data → model → training → prediction logic  

The focus is on building a clear, structured ML pipeline rather than UI complexity.

The model was trained using the Telco Customer Churn dataset available on Kaggle.  
Dataset link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

## 🛠️ Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- HTML / CSS / JavaScript
- Git & GitHub

---

## ✨ Features
- Customer churn prediction using a trained ML model
- Data preprocessing and feature handling
- Model training and evaluation
- Clear separation between training logic and application logic
- End-to-end ML pipeline demonstration

---

## 📂 Project Structure
api/ → backend logic (prediction API)
src/ → model training code
web/ → frontend (HTML, CSS, JavaScript)


---

## 🚀 Running Locally
```bash
pip install -r requirements.txt
python api/app.py
```
## 🧪 The Process

I started by exploring and preprocessing the Telco Customer Churn dataset to understand key features affecting customer behavior.

Next, I trained a machine learning model to predict whether a customer is likely to churn.  
Different approaches were evaluated before selecting the final model.

After training, I separated the model training logic from the application logic to simulate a real-world ML workflow.  
The backend loads the trained model and handles prediction requests.

Throughout the project, I focused on clarity, modularity, and good project structure.

---

## 📚 What I Learned

### 🧠 Machine Learning
- Handling real-world customer data  
- Feature engineering and preprocessing  
- Training and evaluating classification models  

### 🔗 Backend & Integration
- Separating training code from inference logic  
- Loading trained models in an application context  
- Structuring ML projects for maintainability  

### 🌐 Web & Application Logic
- Connecting backend logic with a simple frontend  
- Structuring a lightweight web interface for predictions  

### 🧩 Overall Growth
This project strengthened my understanding of how machine learning models are used beyond notebooks and integrated into real applications.

---

## 🚀 How Can It Be Improved?

- Add more advanced feature engineering  
- Compare multiple models and tuning strategies  
- Improve prediction explainability  
- Enhance the frontend UI/UX  
- Deploy the application for public access  
