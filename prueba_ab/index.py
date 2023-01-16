from flask import Flask, redirect, render_template, request, url_for, flash
import joblib
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
app.secret_key = 'mysecretkey'

import base64

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    print(os.listdir())
    return render_template('index.html')

@app.route('/analisis', methods=['POST'])
def analisis():
    imagen = request.files['img']
    imagen2 = request.files['img2']

    for i in [imagen, imagen2]:
        filename = secure_filename(i.filename)
        i.save(os.path.join(app.config['UPLOAD_FOLDER'] + filename))

    joblib.load("modelo_estimador.pkl")

    flash(f'{type(imagen)} es la mejor imagen para usar')

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(port=3000, debug=True)
