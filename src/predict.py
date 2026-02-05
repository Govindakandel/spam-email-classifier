from src.data_preprocessing import preprocess_text
import pandas as pd



# Predict if a single email is spam or ham
def predict_email(text, model, vectorizer):
     
    # Preprocess email  
    df = preprocess_text(pd.DataFrame({'Message': [text]}), text_column='Message')

    # Convert to vector
    X_vec = vectorizer.transform(df['Message'])

    # Predict
    prediction = model.predict(X_vec)[0]
    return prediction
