import os
from flask import Flask, request, redirect, url_for, render_template
from image_predict import predict_image
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "./static/uploadImage/"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')

@app.route('/img_predict', methods=["POST", "GET"])
def predict():
    f = request.files['file']
    filename = secure_filename(f.filename)
    full_path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(full_path)
    predict_image(full_path)
    return render_template("prediction.html",segFile="static/uploadImage/{}process.jpg".format(filename))


if __name__ == "__main__":
    app.run(debug=True)