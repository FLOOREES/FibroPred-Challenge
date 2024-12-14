from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
import lightgbm as lgb
import numpy as np
import pandas as pd

import shap
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
import lightgbm as lgb
import numpy as np
import pandas as pd

class MedicalAgent:
    def __init__(self, db_path, documents_path='./data/documents'):
        """
        Initializes the medical agent with an LLM, LightGBM models, and a RAG system.

        :param db_path: Path to the user data CSV file.
        :param documents_path: Path to the directory containing documents for the RAG system.
        """
        self.data = Data(db_path)
        self.llm = Ollama(model='llama3.2')
        self.lightgbm_models = self._load_lightgbm_models()
        self.retriever = self._initialize_retriever(documents_path) if documents_path else None

    def _load_lightgbm_models(self):
        """
        Loads LightGBM models from files.

        :return: Dictionary of LightGBM models.
        """
        models = {
            'diabetes_model': lgb.Booster(model_file='diabetes_model.txt'),
            'hypertension_model': lgb.Booster(model_file='hypertension_model.txt')
            # Add other models as needed
        }
        return models

    def _initialize_retriever(self, documents_path):
        """
        Initializes the document retrieval system for RAG.

        :param documents_path: Path to the directory containing documents.
        :return: Document retriever object.
        """
        # Load documents from the specified directory
        loader = DirectoryLoader(
            path=documents_path,
            glob="**/*.txt",  # Adjust the pattern to match your document types
            loader_cls=TextLoader
        )
        documents = loader.load()

        # Generate embeddings for the documents
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(documents, embeddings)

        # Create a retriever from the vector store
        retriever = vector_store.as_retriever()
        return retriever

    def predict_diagnosis(self, user_data, model_name):
        """
        Predicts a diagnosis based on user data using a LightGBM model.

        :param user_data: Dictionary with relevant user information.
        :param model_name: Name of the LightGBM model to use.
        :return: Predicted diagnosis.
        """
        model = self.lightgbm_models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found.")
        data = np.array([user_data[key] for key in sorted(user_data.keys())]).reshape(1, -1)
        diagnosis = model.predict(data)
        return diagnosis[0]

    def explain_diagnosis(self, user_data, model_name):
        """
        Provides a detailed explanation of the diagnosis using SHAP and the LLM.

        :param user_data: Dictionary with relevant user information.
        :param model_name: Name of the LightGBM model used for prediction.
        :return: Explanation of the diagnosis.
        """
        # Predict the diagnosis
        prediction = self.predict_diagnosis(user_data, model_name)

        # Load the corresponding LightGBM model
        model = self.lightgbm_models.get(model_name)
        if not model:
            raise ValueError(f"Model {model_name} not found.")

        # Prepare data for SHAP
        data = np.array([user_data[key] for key in sorted(user_data.keys())]).reshape(1, -1)
        feature_names = sorted(user_data.keys())

        # Initialize SHAP TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)

        # Generate SHAP explanation
        shap_explanation = {}
        for feature, value, shap_value in zip(feature_names, data[0], shap_values[0]):
            shap_explanation[feature] = {
                'value': value,
                'impact': shap_value
            }

        # Retrieve additional information from explicability.txt using RAG
        if self.retriever:
            query = f"Explain the diagnosis {prediction} considering the following shap explanation: {', '.join(feature_names)}."
            additional_info = self.answer_medical_question(query)
        else:
            additional_info = "No additional information available."

        # Combine SHAP explanation with additional information
        explanation = {
            'prediction': prediction,
            'shap_explanation': shap_explanation,
            'additional_info': additional_info
        }

        return explanation

    def answer_medical_question(self, question):
        """
        Answers a medical question using the RAG system.

        :param question: User's medical question.
        :return: Response generated by the RAG system.
        """
        if not self.retriever:
            raise ValueError("Document retrieval system is not initialized.")
        qa_chain = RetrievalQA(llm=self.llm, retriever=self.retriever)
        response = qa_chain.run(question)
        return response

class Data:
    """
    Class to load the user data (csv)
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)