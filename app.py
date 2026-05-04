from flask import Flask, request, jsonify
from model import predict_news, explain_news

app = Flask(__name__)

@app.route("/")
def home():
    return "Fake News Detection API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data["text"]

    result = predict_news(text)
    explanation = explain_news(text, result)

    return jsonify({
        "prediction": result,
        "explanation": explanation
    })

if __name__ == "__main__":
    app.run(debug=True)