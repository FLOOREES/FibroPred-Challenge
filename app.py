from flask import Flask, render_template, request, session, flash, redirect, url_for
import pandas as pd
import os

# Import the MedicalAgent from agent.py
from src.agent import MedicalAgent

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
        return render_template('questionnaire.html')  # Página del cuestionario
    


    ##################### modo CSV #####################

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

            # Ahora que tenemos el filepath, podemos inicializar el MedicalAgent
            # Asumamos que el CSV subido es la base de datos del usuario para el agente
            # Esto cargará los datos y modelos
            agent = MedicalAgent(db_path=filepath, documents_path='./data/documents')

            # Aquí podríamos extraer datos específicos del CSV para la predicción
            # Suponiendo que el CSV tiene columnas esperadas por el modelo, por ejemplo:
            # Digamos que la diabetes_model necesita ['age', 'bmi', 'blood_pressure', ...]
            df = pd.read_csv(filepath)
            
            # Esto es un ejemplo: ajusta las claves según las features que tu modelo espera
            user_data = {}
            # Supongamos que el CSV tiene columnas: 'age', 'bmi', 'blood_pressure'
            # Tomamos la primera fila para el ejemplo
            if not df.empty:
                user_data = {
                    'age': df.iloc[0]['age'],
                    'bmi': df.iloc[0]['bmi'],
                    'blood_pressure': df.iloc[0]['blood_pressure']
                }

            # Predecimos el diagnóstico usando el modelo 'diabetes_model' como ejemplo
            # Asegúrate de que el modelo, las columnas, y el CSV estén alineados
            prediction = agent.predict_diagnosis(user_data, 'diabetes_model')

            # Obtenemos una explicación del diagnóstico
            explanation = agent.explain_diagnosis(user_data, 'diabetes_model')

            # Pasamos el resultado al template results.html
            # results.html podría mostrar la predicción y parte de la explicación
            results_html = render_template(
                'results.html', 
                prediction=prediction, 
                explanation=explanation
            )
            return results_html, 200
        else:
            return "ERROR: El archivo debe ser un CSV.", 400
        

        ####################### modo cuestionario #######################

        @app.route('/submit-questionnaire', methods=['POST'])
        def submit_questionnaire():
            # Obtener los datos enviados por el formulario
            dlco = request.form.get('dlco', type=float)  # DLCO (%) at Diagnosis
            age = request.form.get('age', type=int)  # Age at Diagnosis
            fvc = request.form.get('fvc', type=float)  # FVC (%) at Diagnosis
            final_diagnosis = request.form.get('final_diagnosis', type=int)  # Final Diagnosis
            pirfenidone = request.form.get('pirfenidone')  # Pirfenidone (Binaria: "si" o "no")

            # Convertir el valor binario de Pirfenidone a 0 o 1
            pirfenidone_binary = 1 if pirfenidone == "si" else 0

            # Guardar los datos en variables o procesarlos
            user_data = {
                'dlco': dlco,
                'age': age,
                'fvc': fvc,
                'final_diagnosis': final_diagnosis,
                'pirfenidone': pirfenidone_binary
            }

            # Aquí podrías procesar los datos con tu modelo
            # Por ejemplo:
            # prediction = agent.predict_diagnosis(user_data, 'diabetes_model')

            # Renderizar una página de resultados o redirigir a otra ruta
            return render_template('questionnaire_results.html', user_data=user_data)
    

    return app


app = create_app()

if __name__ == '__main__':
    app.run(debug=True)