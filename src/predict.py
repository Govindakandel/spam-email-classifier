import os
import joblib
from src.data_preprocessing import preprocess_text
import pandas as pd

 # Load trained model and vectorizer from models folder
def load_model_and_vectorizer(model_filename='spam_classifier.pkl', vectorizer_filename='vectorizer.pkl'):

   #  models r directory
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    model_path = os.path.join(models_dir, model_filename)
    vectorizer_path = os.path.join(models_dir, vectorizer_filename)

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    print(f"Model loaded from: {model_path}")
    print(f"Vectorizer loaded from: {vectorizer_path}")
    return model, vectorizer

# Predict if a single email is spam or ham
def predict_email(text, model, vectorizer):
     
    # Preprocess email  
    df = preprocess_text(pd.DataFrame({'Message': [text]}), text_column='Message')

    # Convert to vector
    X_vec = vectorizer.transform(df['Message'])

    # Predict
    prediction = model.predict(X_vec)[0]
    return prediction
