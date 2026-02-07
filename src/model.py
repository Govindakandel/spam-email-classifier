from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import joblib
import os
from  sklearn.metrics import accuracy_score, confusion_matrix , classification_report

 # Train a Multinomial Naive Bayes model
def train_nb(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(solver='liblinear', max_iter=1000)
    model.fit(X_train, y_train)
    return model


#  Save trained model and vectorizer to models directory
def save_model_vectorizer(model, vectorizer, model_filename='spam_classifier.pkl', vectorizer_filename='vectorizer.pkl'):
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, model_filename)
    joblib.dump(model, model_path)
    print(f"Model saved at: {model_path}")

    vectorizer_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(vectorizer_dir, exist_ok=True)

    vectorizer_path = os.path.join(vectorizer_dir, vectorizer_filename)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Vectorizer saved at: {vectorizer_path}")



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


def evaluate_model(model, X_test_vec, y_test):
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", acc)

    # confusion matrix 
    cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])
    print("Confusion Matrix:\n", cm)

    class_report = classification_report(y_test, y_pred, target_names=['ham', 'spam'])
    print("Classification Report:\n", class_report)