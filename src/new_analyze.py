import pandas as pd
from pathlib import Path

# Simple script to check your dataset
file_path = Path(__file__).parent.parent / "data" / "DisasterTweets.csv"
print(f"Checking: {file_path}")

if file_path.exists():
    df = pd.read_csv("C:\\Users\\indhu\\Disaster_ML_Fusion - Copy\\data\\DisasterTweets.csv")
    print("SUCCESS! File loaded.")
    print(f"Columns: {list(df.columns)}")
    print(f"First tweet: {df['Tweets'].iloc[0][:100]}...")
    print(f"Disaster type: {df['Disaster'].iloc[0]}")
else:
    print("FILE NOT FOUND")