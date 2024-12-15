from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.tools import DuckDuckGoSearchRun
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from dotenv import load_dotenv
import os

load_dotenv()

#serpapi_api_key = os.getenv("SERPAPI_API_KEY")

class MedicalAgent:
    def __init__(self, db_path, documents_path='./documents', latent=False):
        """
        Initializes the medical agent with an LLM, LightGBM models, a RAG system, and internet search capability.

        :param db_path: Path to the user data CSV file.
        :param documents_path: Path to the directory containing documents for the RAG system.
        """
        self.data = Data(db_path).data

        self.year = self._get_year()

        self.latent = latent

        self.llm = Ollama(model='llama3.2')

        self.model = self._load_models()

        self.retriever = self._initialize_retriever(documents_path) if documents_path else None

        self.search_tool = self._initialize_search_tool()

        self.agent = self._initialize_agent()

    def _get_year(self):
        """
        Determines the year based on the number of columns in the data.

        :return: None
        """
        num_columns = self.data.shape[1] 
        if num_columns == 49:return 2
        elif num_columns == 48:return 1
        else:return 0

    def _load_models(self):
        """
        Loads LightGBM models from files.

        :return: Dictionary of LightGBM models.
        """
        assert os.path.exists('models/death_model_y0.txt'), "El archivo death_model_y0.txt no existe."
        assert os.path.exists('models/prog_model_y0.txt'), "El archivo prog_model_y0.txt no existe."
        b1 = lgb.Booster(model_file='src/model_weights/death_model_y0.txt')
        b2 = lgb.Booster(model_file='src/model_weights/prog_model_y0.txt')
        print('juan')
        models = {
            'year0': [b1,b2]
        }

        if self.year == 0 and self.latent == False:
            print('.........................................................................................................')
            return models['year0']

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

    def _initialize_search_tool(self):
        """
        Initializes the internet search tool using DuckDuckGo.
        """
        search = DuckDuckGoSearchRun()
        tool = Tool(
            name="duckduckgo_search",
            description="Busca en internet información actualizada.",
            func=search.run,
        )
        return tool

    def _initialize_agent(self):
        """
        Initializes the agent with the LLM and tools.
        """
        tools = [self.search_tool]
        agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        return agent

    def predict_diagnosis(self):
        """
        Predicts a diagnosis based on the current data in self.data using the appropriate LightGBM model.

        :return: Predicted diagnosis.
        """
        if not self.model:
            raise ValueError("No model is loaded for the current year and latent state.")

        # Ensure that self.data contains exactly one row of data
        if self.data.shape[0] != 1:
            raise ValueError("The data should contain exactly one row for prediction.")

        # Prepare the data for prediction
        self.X = self.data.values  # Convert the DataFrame row to a NumPy array
        self.death_model = self.model[0]
        self.prog_model = self.model[1]
        diagnosis = self.death_model.predict(self.X)
        prognosis = self.prog_model.predict(self.X)

        # Return the first (and only) prediction
        return diagnosis, prognosis

    def explain_diagnosis(self):
        """
        Provides a detailed explanation of the diagnosis using SHAP and the LLM.

        :param user_data: Dictionary with relevant user information.
        :param model_name: Name of the LightGBM model used for prediction.
        :return: Explanation of the diagnosis.
        """
        # Predict the diagnosis
        death_prediction, progressive_prediction = self.predict_diagnosis()

        # Prepare data for SHAP
        if not self.model:
            raise ValueError("No model is loaded for SHAP explanations.")

        # Initialize SHAP TreeExplainers for both models
        death_explainer = shap.TreeExplainer(self.death_model)
        prog_explainer = shap.TreeExplainer(self.prog_model)

        # Compute SHAP values
        death_shap_values = death_explainer.shap_values(self.X)
        prog_shap_values = prog_explainer.shap_values(self.X)

        # Generate SHAP explanations for the diagnosis model
        feature_names = self.data.columns
        death_shap_explanation = {
            feature: {
                'value': self.X[0][i],
                'impact': death_shap_values[0][i]
            }
            for i, feature in enumerate(feature_names)
        }

        # Generate SHAP explanations for the prognosis model
        prog_shap_explanation = {
            feature: {
                'value': self.X[0][i],
                'impact': prog_shap_values[0][i]
            }
            for i, feature in enumerate(feature_names)
        }

        # Retrieve additional information using RAG
        if self.retriever:
            query = (
                f"Explain the death prediction ({death_prediction}) and prognosis ({progressive_prediction}) "
                f"considering the following SHAP explanations for the features: {', '.join(feature_names)}."
            )
            additional_info = self.answer_medical_question(query)

        # Combine SHAP explanations and additional information into the output
        explanation = {
            'death_prediction': death_prediction,
            'prog_prediction': progressive_prediction,
            'death_shap_explanation': death_shap_explanation,
            'prog_shap_explanation': prog_shap_explanation,
            'additional_info': additional_info
        }

        return explanation

    def answer_medical_question(self, question):
        """
        Answers a medical question using the RAG system and internet search.

        :param question: User's medical question.
        :return: Response generated by the agent.
        """
        if not self.retriever:
            raise ValueError("Document retrieval system is not initialized.")
        qa_chain = RetrievalQA(llm=self.llm, retriever=self.retriever)
        response = qa_chain.run(question)

        # If the retriever doesn't provide a satisfactory answer, use the internet search tool
        if not response or "I don't know" in response:
            response = self.search_tool.run(question)

        return response

class Data:
    """
    Class to load the user data (CSV).
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)