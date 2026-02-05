import pandas as pd
import os 

# read data from csv file
def load_data(path):
    df = pd.read_csv(path)
    print(f"Data loaded from: {path}")
    return df

# text preprocessing function
def preprocess_text(df, text_column='Message'):
    
    # drop null values
    df = df.dropna(subset=[text_column])


    # text preprocessing 
    df[text_column] = df[text_column].str.lower()
    df[text_column] = df[text_column].str.replace(r'\W', ' ', regex=True)
    df[text_column] = df[text_column].str.replace(r'\d', '', regex=True)
    df[text_column] = df[text_column].str.strip()

    # duplicate removal
    df = df.drop_duplicates()


    return df

