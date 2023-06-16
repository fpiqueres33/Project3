from flask import Flask, request, render_template, send_file, make_response
from werkzeug.utils import secure_filename
import os
from main import summarize
from Analytics import analyze_text, plot_histogram
import json
from Analytics import get_text_statistics, generate_histogram

app = Flask(__name__)

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

    summary = summarize(file_path, percentile)
    return render_template('index.html', **summary)


@app.route('/api/analyze', methods=['POST'])
def handle_analysis():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join('uploads', filename)
    file.save(file_path)

    file_content = read_file(file_path)
    analysis = get_text_statistics(file_content)
    histogram_path = generate_histogram(file_content, 'static/histogram.png')

    return render_template('analysis.html', analysis=analysis, histogram=histogram_path)

if __name__ == '__main__':
    app.run(debug=True)
