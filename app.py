from flask import Flask, render_template, request, session, flash, redirect, url_for
import pandas as pd
import os

file_dir = os.path.dirname(__file__)
os.chdir(file_dir)

def create_app():
    app = Flask(__name__)

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
            flash('No se seleccionó ningún archivo.', 'danger')
            return redirect(url_for('upload_csv'))

        file = request.files['csvFile']

        if file.filename == '':
            flash('El archivo no tiene nombre.', 'danger')
            return redirect(url_for('upload_csv'))

        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)  # Guarda el archivo en el servidor

            # Aquí redirigimos al template de carga
            return render_template('loading.html', filename=filepath)
        else:
            flash('El archivo debe ser un CSV.', 'danger')
            return redirect(url_for('upload_csv'))

    @app.route('/process', methods=['POST'])
    def process():
        filepath = request.form.get('filename')

        if not filepath or not os.path.exists(filepath):
            flash('No se pudo encontrar el archivo para procesar.', 'danger')
            return redirect(url_for('upload_csv'))

        # Leer el CSV y procesarlo
        df = pd.read_csv(filepath)
        flash(f'Archivo procesado exitosamente. Contiene {len(df)} filas y {len(df.columns)} columnas.', 'success')
        
        # Aquí podrías pasar el DataFrame a un modelo o realizar más acciones
        print(df.head())
        
        return redirect(url_for('upload_csv'))

    return app




app = create_app()

if __name__ == '__main__':
    app.run(debug=True)