from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from graph_builder import graph

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    user_input = request.form["user_input"]
    input_state = {"input": user_input}

    final_state = graph.invoke(input_state)

    return jsonify({
        "emotion": final_state.get("emotions"),
        "forecast": final_state.get("forecast"),
        "emotion_context_links": final_state.get("emotion_context_links"),
        "self_care_articles": final_state.get("self_care_articles"),
        "rag_self_care": final_state.get("rag_self_care"),
        "prompt": final_state.get("tailored_prompt"),
        "care_suggestion": final_state.get("care_suggestion"),
        "appointment_offer": final_state.get("appointment_offer"),
        "appointment_status": final_state.get("appointment_status"),
        "therapist_match": final_state.get("matched_therapist_rag"),
        "agent_router_output": final_state.get("agent_router_output"),
        "router_trace": final_state.get("router_trace"),
        "crisis_response": final_state.get("crisis_response"),
    })

if __name__ == "__main__":
    app.run(debug=True)
    

