from flask import Flask, render_template, request, session, flash, redirect, url_for
import pandas as pd
import os

file_dir = os.path.dirname(__file__)
os.chdir(file_dir)

def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = os.path.join(file_dir, 'uploads')
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    @app.route('/')
    def titulo():
        return render_template('titulo.html')
    
    @app.route('/diagnosis-questionnaire')
    def questionnaire():
        return render_template('questionnaire.html')  # Página para el cuestionario

    @app.route('/upload-csv', methods=['GET'])
    def upload_csv():
        return render_template('upload_csv.html')

    @app.route('/load', methods=['POST'])
    def load():
        if 'csvFile' not in request.files:
            return "ERROR: No se seleccionó ningún archivo.", 400

        file = request.files['csvFile']

        if file.filename == '':
            return "ERROR: El archivo no tiene nombre.", 400

        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)  # Guarda el archivo en el servidor

            # Procesar el archivo CSV
            df = pd.read_csv(filepath)
            # Aquí podrías hacer más procesamiento o análisis

            # Retornamos el HTML de results
            results_html = render_template('results.html')
            return results_html, 200
        else:
            return "ERROR: El archivo debe ser un CSV.", 400

    return app


app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
