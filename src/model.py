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

#  Load trained model from models directory
def load_model(filename='spam_classifier.pkl'):
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    path = os.path.join(models_dir, filename)
    model = joblib.load(path)
    print(f"Model loaded from: {path}")
    return model


def save_vectorizer(vectorizer, filename='vectorizer.pkl'):
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)

    path = os.path.join(models_dir, filename)
    joblib.dump(vectorizer, path)
    print(f"Vectorizer saved at: {path}")