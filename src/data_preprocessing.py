import pandas as pd

def load_data(path="./data/raw/spam.csv"):
    df = pd.read_csv(path)
    return df

def preprocess_text(df, text_column='Message'):
    # drop null values
    print("Dropping null values...")
    print(f"Null value count before dropping: {df[text_column].isnull().sum()}")
    print(f"Shape before dropping null values: {df.shape}")
    df = df.dropna(subset=[text_column])
    print(f"Shape after dropping null values: {df.shape}")

    # text preprocessing 
    df[text_column] = df[text_column].str.lower()
    df[text_column] = df[text_column].str.replace(r'\W', ' ', regex=True)
    df[text_column] = df[text_column].str.replace(r'\d', '', regex=True)
    df[text_column] = df[text_column].str.strip()

    # duplicate removal
    print("Removing duplicates...")
    print(f"Duplicate count: {df.duplicated().sum()}")
    print(f"Shape before removing duplicates: {df.shape}")
    df = df.drop_duplicates()
    print(f"Shape after removing duplicates: {df.shape}")

    return df

