from flask import Flask, render_template, request, session
import os

def create_app():
    app = Flask(__name__)

    @app.route('/')
    def titulo():
        
        return render_template('templates/titulo.html')
    
    return app




app = create_app()

if __name__ == '__main__':
    app.run(debug=True)