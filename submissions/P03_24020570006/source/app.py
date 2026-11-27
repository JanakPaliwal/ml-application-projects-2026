import os
import json
from flask import Flask, render_template, jsonify, request
import subprocess

app = Flask(__name__)
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(DATA_DIR, "forecasting_results.json")

def get_forecast_data():
    if not os.path.exists(RESULTS_FILE):
        return None
    with open(RESULTS_FILE, "r") as f:
        return json.load(f)

@app.route("/")
def index():
    data = get_forecast_data()
    return render_template("index.html", data=data)

@app.route("/api/forecast/<crop>")
def api_forecast(crop):
    data = get_forecast_data()
    if not data or crop.lower() not in data:
        return jsonify({"error": "Crop not found or data missing"}), 404
    return jsonify(data[crop.lower()])

@app.route("/api/retrain", methods=["POST"])
def retrain():
    try:
        pipeline_path = os.path.join(DATA_DIR, "ml_pipeline.py")
        result = subprocess.run(["python", pipeline_path], capture_output=True, text=True, check=True)
        return jsonify({"status": "success", "message": "Models retrained successfully!", "output": result.stdout})
    except subprocess.CalledProcessError as e:
        return jsonify({"status": "error", "message": str(e), "stderr": e.stderr}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
