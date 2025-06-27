from flask import Flask, request, jsonify, render_template
from router import build_graph

app = Flask(__name__)
emotion_graph = build_graph()

@app.route("/")
def home():
    return render_template("index.html")  # You can create a simple HTML UI

@app.route("/analyze", methods=["POST"])
def analyze_emotion():
    user_input = request.json.get("text", "")
    if not user_input:
        return jsonify({"error": "No text provided"}), 400

    # Run LangGraph with user input
    result = emotion_graph.invoke({"input": user_input})
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
