from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"  # Define the folder to save uploaded images
os.makedirs(
    app.config["UPLOAD_FOLDER"], exist_ok=True
)  # Create the folder if it does not exist


@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files["file"]
    theme = request.form.get("theme", "")


if __name__ == "__main__":
    app.run(debug=True)
