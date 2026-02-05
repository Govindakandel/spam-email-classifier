from src.data_preprocessing import load_data, preprocess_text
from src.feature_extraction import get_bow_features
from src.model import train_nb, save_model , save_vectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import os


def train_model(csv_path):
    print(csv_path)
    # load and preprocess data
    df = load_data(path=csv_path)
    df = preprocess_text(df)

    #  Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['Message'], df['Category'], test_size=0.3, random_state=42
    )

    # Feature extraction
    X_train_vec, X_test_vec, vectorizer = get_bow_features(X_train, X_test)

    #  Train model
    model = train_nb(X_train_vec, y_train)

    #  Evaluate
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", acc)

    # confusion matrix 
    cm = confusion_matrix(y_test, y_pred, labels=['ham', 'spam'])
    print("Confusion Matrix:\n", cm)

    # Save model
    save_model(model)

    # Save vectorizer / bags of words
    save_vectorizer(vectorizer)

if __name__ == "__main__":
   train_model(csv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'spam_processed.csv'))
