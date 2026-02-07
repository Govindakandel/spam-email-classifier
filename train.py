from src.data_preprocessing import load_data, preprocess_text
from src.feature_extraction import get_bow_features , get_tfidf_features
from src.model import train_nb, train_logistic_regression,  save_model_vectorizer, evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix , classification_report
import os




def train_model_nb(X_train, y_train, X_test, y_test) :

    # Feature extraction
    X_train_vec, X_test_vec, vectorizer = get_bow_features(X_train, X_test)

    #  Train model
    model = train_nb(X_train_vec, y_train)
    
    return model, vectorizer

   


def train_model_logistic_regression(X_train, y_train, X_test, y_test) :
    # Feature extraction
    X_train_vec, X_test_vec, vectorizer = get_tfidf_features(X_train, X_test)

    #  Train model
    model = train_logistic_regression(X_train_vec, y_train)
    return model, vectorizer


def train(csv_path) :
    # load and preprocess data
    df = load_data(path=csv_path)
    df = preprocess_text(df)

    #  Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['Message'], df['Category'], test_size=0.3, random_state=42
    )
    # Train Naive Bayes model
    print("Training Naive Bayes model...")
    print("-" * 30)
    model_nb , vectorizer_nb =  train_model_nb(X_train, y_train, X_test, y_test)
    print("\nNaive Bayes model trained successfully!\n")

    # Evaluate Naive Bayes model
    print("Evaluating Naive Bayes model...")
    print("-" * 30)
    X_test_vec_nb = vectorizer_nb.transform(X_test)
    evaluate_model(model_nb, X_test_vec_nb, y_test)
    print("-" * 30)
 
 
    # Train Logistic Regression model
    print("Training Logistic Regression model...")
    print("-" * 30)
    model_logistic , vectorizer_logistic = train_model_logistic_regression(X_train, y_train, X_test, y_test)
    print("\nLogistic Regression model trained successfully!\n")
    print("-" * 30)

    # Evaluate Logistic Regression model
    print("Evaluating Logistic Regression model...")
    print("-" * 30)
    X_test_vec_logistic = vectorizer_logistic.transform(X_test)
    evaluate_model(model_logistic, X_test_vec_logistic, y_test)
    print("-" * 30)

    # Save Naive Bayes model and vectorizer
    save_model_vectorizer(model_nb, vectorizer_nb, model_filename='nb_spam_classifier.pkl', vectorizer_filename='nb_vectorizer.pkl')
    # Save Logistic Regression model and vectorizer
    save_model_vectorizer(model_logistic, vectorizer_logistic, model_filename='logistic_spam_classifier.pkl', vectorizer_filename='logistic_vectorizer.pkl')    


if __name__ == "__main__":
   train(csv_path=os.path.join(os.getcwd(), 'data', 'spam.csv'))
