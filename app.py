from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Define the folder to save uploaded images
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Create the folder if it does not exist

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    theme = request.form.get('theme', '')

    # If the user does not select a file, the browser submits an empty part without filename
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({"message": "File uploaded successfully", "theme": theme}), 200

if __name__ == '__main__':
    app.run(debug=True)
