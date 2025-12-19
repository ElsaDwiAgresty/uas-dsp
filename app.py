from flask import Flask, render_template, request
import os
from model_util import predict_image

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard_view.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        file = request.files["file"]
        filename = file.filename
        path = os.path.join("static/uploads", filename)
        file.save(path)

        result = predict_image(path)

        return render_template(
            "result.html",
            image=filename,
            genus=result["genus"],
            edibility=result["edibility"]
        )

    return render_template("predict.html")

if __name__ == "__main__":
    app.run(debug=True)