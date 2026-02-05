
# feature_extraction text to features using BoW 
from sklearn.feature_extraction.text import CountVectorizer

def get_bow_features(X_train, X_test):

    # Converts text to Bag of Words features

    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer
