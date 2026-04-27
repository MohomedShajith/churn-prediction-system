import os
import torch
import joblib
import pandas as pd
from torch import nn
from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
from model.train import ChurnModel
from sklearn.preprocessing import StandardScaler

load_dotenv()

url = os.getenv("MONGO_URI")
client = MongoClient(url)
db = client["churndb"]
collection = db["predictions"]



class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

EXPECTED_COLUMNS = ['gender', 'SeniorCitizen', 'Partner',
 'Dependents', 'tenure', 'PhoneService', 'PaperlessBilling',
 'MonthlyCharges', 'TotalCharges',  'MultipleLines_No phone service',
 'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No',
 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
 'OnlineBackup_No internet service', 'OnlineBackup_Yes',
 'DeviceProtection_No internet service', 'DeviceProtection_Yes',
 'TechSupport_No internet service', 'TechSupport_Yes',
 'StreamingTV_No internet service', 'StreamingTV_Yes',
 'StreamingMovies_No internet service', 'StreamingMovies_Yes',
 'Contract_One year', 'Contract_Two year',
 'PaymentMethod_Credit card (automatic)',
 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

scaler = joblib.load("model/scaler.pkl")
model = ChurnModel()
model.load_state_dict(torch.load("model/churn_model.pth", weights_only=True))
model.eval()



app = FastAPI()
@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict(customer: Customer):
    df = pd.DataFrame([customer.dict()])
    df['gender'] = df['gender'].map({'Male':1,'Female':0})
    cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for i in cols:
        df[i] = df[i].map({'Yes':1,'No':0})
    hotcols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaymentMethod']

    df = pd.get_dummies(df, columns=hotcols, drop_first=True)
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = 0
            
    df = df[EXPECTED_COLUMNS]

    df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])

    df = df.values
    df = torch.FloatTensor(df)

    pred_logist = model(df)
    preds = torch.round(pred_logist)
    collection.insert_one({"customer":customer.dict(),
    "predictions":int(preds.item()),
    "probability":round(pred_logist.item(),4),
    "timestamp":datetime.utcnow()})

    return {"churn_prediction": int(preds.item()),"churn_probability": round(pred_logist.item(),4) }


@app.get("/history")
def history():
   results = collection.find({}, {"_id": 0})
   return list(results)


    




