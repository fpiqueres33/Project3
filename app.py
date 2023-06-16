from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
from main import summarize

app = Flask(__name__)

#Creamos el directorio upload para guardar los archivos que seleccionamos
UPLOAD_FOLDER = 'uploads'
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/api/summarize', methods=['POST'])
def handle_form():
    file = request.files['file']
    percentile = request.form.get('percentile')
    filename = secure_filename(file.filename)
    file_path = os.path.join('uploads', filename)
    file.save(file_path)

    # Convert percentile to integer
    try:
        percentile = int(percentile)
    except ValueError:
        percentile = 80  # Default value if conversion fails

    with open(file_path, "r", encoding='utf-8') as file:
        file_content = file.read()

    #summary = summarize(file_content, percentile)
    summary = summarize(os.path.join('uploads', file_content), percentile)
    return render_template('index.html', summary=summary)





if __name__ == '__main__':
    app.run(debug=True)
