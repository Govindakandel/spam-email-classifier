from sklearn.naive_bayes import MultinomialNB
import joblib

 # Train a Multinomial Naive Bayes model
def train_nb(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

#  Save trained model to disk
def save_model(model, path='../models/spam_classifier.pkl'):
    joblib.dump(model, path)

#  Load trained model from disk
def load_model(path='../models/spam_classifier.pkl'):
    return joblib.load(path)
