import sys
from src.predict import predict_email
from src.model import load_model_and_vectorizer

def main():
 
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"email text\"")
        return

    text = sys.argv[1]

    model, vectorizer = load_model_and_vectorizer()
    prediction = predict_email(text, model, vectorizer)

    print("Prediction:", prediction)

if __name__ == "__main__":
    main()
