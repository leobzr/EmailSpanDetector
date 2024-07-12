import joblib
import sys

def load_model():
    model_path = "C:\\Users\\Leo\\Desktop\\Python Big projects\\EmailSpanDetector\\spam_model.pkl"
    vectorizer_path = "C:\\Users\\Leo\\Desktop\\Python Big projects\\EmailSpanDetector\\spam_vectorizer.pkl"
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict(text):
    model, vectorizer = load_model()
    text_transformed = vectorizer.transform([text])
    prediction = model.predict(text_transformed)
    return "Spam" if prediction[0] == 1 else "Not Spam"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = sys.argv[1]
        prediction = predict(text)
        print(f"The message is {prediction}")
    else:
        print("Please provide a text message as a command-line argument.")
