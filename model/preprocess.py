import pandas as pd
from sqlalchemy import create_engine
import torch
from torch import nn
import os
import joblib
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def get_data():

    load_dotenv()

    db_url = f'postgresql://{os.getenv("DB_USER")}:{os.getenv("DB_PASSWORD")}@{os.getenv("DB_HOST")}:{os.getenv("DB_PORT")}/{os.getenv("DB_NAME")}'

    engine = create_engine(db_url)

    query = "select * from customers;"

    data = pd.read_sql(query, engine)




    data = data.drop('customerID',axis=1)


    data['gender'] = data['gender'].map({'Male':1,'Female':0})
    cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for i in cols:
        data[i] = data[i].map({'Yes':1,'No':0})



    hotcols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']

    data = pd.get_dummies(data, columns=hotcols, drop_first=True)




    bool_cols = data.select_dtypes(include='bool').columns
    data[bool_cols] = data[bool_cols].astype(int)

    #print(data)

    #print(data.dtypes)


    sc = StandardScaler()

    sc_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    data[sc_cols] = sc.fit_transform(data[sc_cols])

    #print(data.columns)


    X = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state = 42)



    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)


    joblib.dump(sc, "model/scaler.pkl")
    return X_train, X_test, y_train, y_test

    print(data.dtype)