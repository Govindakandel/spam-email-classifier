from sklearn.naive_bayes import MultinomialNB
import joblib
import os

 # Train a Multinomial Naive Bayes model
def train_nb(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

#  Save trained model to models directory
def save_model(model, filename='spam_classifier.pkl'):
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)

    path = os.path.join(models_dir, filename)
    joblib.dump(model, path)
    print(f"Model saved at: {path}")

#cSave vectorizer to models directory
def save_vectorizer(vectorizer, filename='vectorizer.pkl'):
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)

    path = os.path.join(models_dir, filename)
    joblib.dump(vectorizer, path)
    print(f"Vectorizer saved at: {path}")



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