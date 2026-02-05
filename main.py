from src.data_preprocessing import load_data, preprocess_text   


def main():
    # load data
    df = load_data()
    print("Data loaded successfully.")

    # preprocess text
    df = preprocess_text(df)
    print("Text preprocessing completed.")

if __name__ == "__main__":
    main()