from dotenv import load_dotenv
import pandas as pd
import os
from sqlalchemy import create_engine

load_dotenv()

db_url = f"postgresql://{os.getenv("DB_USER")}:{os.getenv("DB_PASSWORD")}@{os.getenv("DB_HOST")}:{os.getenv("DB_PORT")}/{os.getenv("DB_NAME")}"

engine = create_engine(db_url)

churn_data = pd.read_csv(r"data\telco_churn.csv")



#print(churn_data[churn_data["TotalCharges"] == " "])

churn_data = churn_data[~(churn_data["TotalCharges"] == " ")]
print(len(churn_data))

churn_data["TotalCharges"] = pd.to_numeric(churn_data["TotalCharges"])

churn_data["Churn"] = churn_data["Churn"].map({"Yes": 1, "No": 0})

print(churn_data.dtypes)

churn_data.to_sql("customers", engine, if_exists="replace", index=False)
