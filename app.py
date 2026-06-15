from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load Model and Encoders
model = joblib.load("models/income_model.pkl")
encoders = joblib.load("models/encoder.pkl")

print("Model Loaded Successfully!")

try:
    print("Model Features:")
    print(model.feature_names_in_)
except Exception:
    print("feature_names_in_ not available")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/analytics", methods=["GET"])
def analytics():
    import json
    import os

    metrics_path = "models/analytics_metrics.json"

    if not os.path.exists(metrics_path):
        return jsonify({"error": "analytics_metrics.json not found. Run train_model.py first."}), 500

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return jsonify(metrics)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        print("\nReceived Data:")
        print(data)

        # NOTE: the trained model expects the exact feature columns it saw during fit.
        # In this project those are:
        # ['age','workclass','fnlwgt','education','education.num','marital.status',
        #  'occupation','relationship','race','sex','capital.gain','capital.loss',
        #  'hours.per.week','native.country']
        model_features = list(getattr(model, "feature_names_in_", []))

        input_row = {
            "age": int(data["age"]),
            "workclass": data["workclass"],

            # map UI fields -> expected training field names
            "education.num": int(data["education_num"]),
            "marital.status": data["marital_status"],
            "occupation": data["occupation"],
            "sex": data["sex"],
            "capital.gain": int(data["capital_gain"]),
            "capital.loss": int(data["capital_loss"]),
            "hours.per.week": int(data["hours_per_week"]),

            # provide reasonable defaults for columns not collected in the UI
            "fnlwgt": 0,
            "education": "Bachelors",
            "relationship": "Not-in-family",
            "race": "White",
            "native.country": "United-States",
        }

        input_data = pd.DataFrame([input_row])

        # Reindex to match model feature order exactly.
        input_data = input_data.reindex(columns=model_features)

        print("\nBefore Encoding:")
        print(input_data)

        # Encode categorical columns used during training.
        # Train used LabelEncoder for every object column, and stored them in encoders.
        encoded_cols = []

        for col in model_features:
            if col not in input_data.columns:
                continue
            if col not in encoders:
                continue

            encoder = encoders[col]
            try:
                input_data[col] = encoder.transform(input_data[col].astype(str))
                encoded_cols.append(col)
            except Exception as e:
                return jsonify({
                    "error": f"Encoding failed for '{col}' with value '{input_data.at[0, col]}': {str(e)}",
                    "hint": "The provided input category is not present in training data for that feature."
                }), 400

        # Validate: after encoding, all model features must be numeric.
        non_numeric_cols = [
            c for c in model_features
            if c in input_data.columns and not pd.api.types.is_numeric_dtype(input_data[c])
        ]
        if non_numeric_cols:
            return jsonify({
                "error": "Some model features are still non-numeric after encoding.",
                "non_numeric_cols": non_numeric_cols,
                "encoded_cols": encoded_cols,
                "missing_encoder_columns": [c for c in non_numeric_cols if c not in encoders]
            }), 400

        print("\nAfter Encoding:")
        print(input_data)

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        confidence = round(float(max(probability)) * 100, 2)

        result = (
            "Income > 50K"
            if prediction == 1
            else "Income ≤ 50K"
        )

        print("\nPrediction:", result)
        print("Confidence:", confidence)

        return jsonify({
            "prediction": result,
            "confidence": confidence
        })

    except Exception as e:
        print("\nERROR:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

