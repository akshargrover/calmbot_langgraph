from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from graph_builder import analyze_graph
from utils.data_schema import AnalyzeResponse

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    user_id = data.get("user_id", "default")
    text = data["text"]
    state = {"user_id": user_id, "text": text}
    result = analyze_graph.invoke(state)
    response = AnalyzeResponse(
        emotions=result.get("emotions", ""),
        confidence=result.get("confidence", 0.0),
        suggestion=result.get("suggestion", ""),
        tailored_prompt=result.get("tailored_prompt", ""),
        forecast=result.get("forecast", ""),
        similar_past_moods=result.get("similar_past_moods", [])
    )
    return jsonify(response.__dict__)

if __name__ == "__main__":
    app.run(debug=True)
