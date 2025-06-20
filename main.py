from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

app = FastAPI()

model = joblib.load("xgboost_podcast_listening_time_predictor.pkl")

class PredictionRequest(BaseModel):
    Podcast_Name: str
    Episode_Title: str
    Episode_Length_minutes: float
    Genre: str
    Host_Popularity_percentage: float
    Publication_Day: str
    Publication_Time: str
    Guest_Popularity_percentage: float | None
    Number_of_Ads: int
    Episode_Sentiment: str

def feature_engineering(df_tmp):
    df = df_tmp.copy()
    print(df)
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    df = df[
        (df['Host_Popularity_percentage'] >= 0) & (df['Host_Popularity_percentage'] <= 100) &
        (df['Guest_Popularity_percentage'] >= 0) & (df['Guest_Popularity_percentage'] <= 100) &
        (df['Number_of_Ads'].isin([0, 1, 2, 3]))
    ]

    df['Episode_Number'] = df['Episode_Title'].str.extract(r'Episode (\d+)', expand=False).fillna(0)
    df['Episode_Number'] = pd.to_numeric(df['Episode_Number'], errors='coerce').astype(int)

    df['Guest_Popularity_missing'] = df['Guest_Popularity_percentage'].isna().astype(int)
    df['Guest_Popularity_percentage'].fillna(0, inplace=True)

    df["Number_of_Ads"].fillna(1, inplace=True)
    df['Episode_Length_minutes'] = pd.to_numeric(df['Episode_Length_minutes'], errors='coerce')
    df['Host_Popularity_percentage'] = pd.to_numeric(df['Host_Popularity_percentage'], errors='coerce')
    
    df['Guest_Popularity_percentage'] = pd.to_numeric(df['Guest_Popularity_percentage'], errors='coerce')
    df['Number_of_Ads'] = df['Number_of_Ads'].astype(int)
    

    day_order = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 
                 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    df['Publication_Day_Numeric'] = df['Publication_Day'].map(day_order).fillna(0)

    time_order = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
    df['Publication_Time_Numeric'] = df['Publication_Time'].map(time_order).fillna(0)

    sentiment_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    df['Episode_Sentiment_Numeric'] = df['Episode_Sentiment'].map(sentiment_map).fillna(0)

    df['Ads_per_minute'] = df['Number_of_Ads'] / (1 + df['Episode_Length_minutes'])
    df['Has_Ads'] = (df['Number_of_Ads'] > 0).astype(int)

    df['Is_Weekend'] = df['Publication_Day_Numeric'].isin([5, 6]).astype(int)

    df['Medium_Episode_Length'] = ((df['Episode_Length_minutes'] > 60) & (df['Episode_Length_minutes'] <= 90)).astype(int)
    df['Hight_Episode_Length'] = (df['Episode_Length_minutes'] > 90).astype(int)

    df['Hight_Host_Popularity'] = (df['Host_Popularity_percentage'] > 80).astype(int)
    df['Hight_Guest_Popularity'] = (df['Guest_Popularity_percentage'] > 80).astype(int)

    df["Guest_Host_Combined_Popularity"] = (df["Guest_Popularity_percentage"] + df["Host_Popularity_percentage"]) / 2

    df = df.drop(columns=[
        "Publication_Day", "Publication_Time", "Episode_Sentiment", "Episode_Title"
    ])

    return df

@app.post("/predict")
def predict(request: PredictionRequest):
    input_df = pd.DataFrame([request.dict()])
    processed_df = feature_engineering(input_df)
    prediction = model.predict(processed_df)
    return {"prediction": float(prediction[0])}
