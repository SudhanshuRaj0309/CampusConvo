from flask import Flask, request, jsonify, render_template
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.schema import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from googletrans import Translator
import random
import os

os.environ['OPENAI_API_KEY'] =  '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'
app = Flask(__name__)


general_questions = {
    "hi": [
        "Hello! Welcome to the MNIT chatbot. How may I assist you today?",
        "Hi, I'm here to help you with any questions about MNIT.",
        "Greetings! I am the MNIT chatbot. Feel free to ask anything about MNIT."
    ],
    "hello": [
        "Hi! How can I assist you today?",
        "Hi! How can I be of service to you?",
        "Hello! I'm here to provide information and answer your queries about MNIT."
    ],
    "how are you": [
        "I'm an AI, so I don't have feelings, but I'm here to help you with any questions about MNIT.",
        "As an AI chatbot, I'm always ready to assist you with information about MNIT.",
        "I'm here to provide you with answers and information about MNIT. How can I assist you today?"
    ],
    "who are you": [
        "I am a chatbot dedicated to providing information and answering queries about MNIT.",
        "I am an AI assistant designed to help you explore the teachings and concepts of MNIT.",
        "I am the virtual assistant for MNIT, here to assist you on your campus journey."
    ],
    "what is your purpose": [
        "My purpose is to provide information and answer questions about MNIT.",
        "I'm here to help you explore the MNIT campus, academics, and diffeeerent branches in the MNIT.",
        "I am designed to assist you in understanding MNIT."
    ],
    "tell me about yourself": [
        "I am an AI chatbot dedicated to providing information and answering questions about MNIT.",
        "I am the virtual assistant for MNIT, here to assist you on your campus journey.",
        "I am programmed to help you explore the MNIT."
    ],
    "what can you do": [
        "I can provide information and answer questions about MNIT.",
        "I am here to assist you in understanding the MNIT.",
        "I can help you explore the MNIT and address your queries about it."
    ],
    "where are you from": [
        "I am a virtual assistant created to provide information and answer questions about MNIT.",
        "I exist to assist you in exploring the MNIT.",
        "I am a digital companion focused on helping you learn about MNIT."
    ],
    "what is your favorite color": [
        "As an AI, I don't have personal preferences, but I'm here to help you with any questions about MNIT.",
        "I don't have a favorite color, but I'm dedicated to providing information about MNIT.",
        "I'm more interested in answering your queries about MNIT than discussing colors."
    ],
    "do you have any hobbies": [
        "As an AI, I don't have personal hobbies. My purpose is to assist you with questions about MNIT.",
        "I don't have hobbies, but I'm here to support your exploration of MNIT.",
        "I'm focused on providing information and insights related to MNIT rather than hobbies."
    ],
    "what is your favorite book": [
        "As an AI chatbot, I don't have personal preferences. But I can provide information about MNIT.",
        "I don't have a favorite book, but I'm here to assist you in exploring the MNIT.",
        "My focus is on MNIT."
    ]
    # Your general questions and responses here...
}

questions = list(general_questions.keys())
answers = [answer for answer_list in general_questions.values() for answer in answer_list]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

knn = NearestNeighbors(n_neighbors=1)
knn.fit(question_vectors)


class Genie:

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.loader = UnstructuredFileLoader(self.file_path)
        self.documents = self.loader.load()
        self.texts = self.text_split(self.documents)
        self.vectordb = self.embeddings(self.texts)
        self.fallback = self.generate_response
        llm = OpenAI(temperature=0.3)
        self.genie = load_qa_chain(llm, chain_type="stuff")

    def generate_response(self, query: str):
        query_vector = vectorizer.transform([query])
        _, closest_index = knn.kneighbors(query_vector)
        closest_question = questions[closest_index[0][0]]
        if closest_question in general_questions:
            return general_questions[closest_question]

    @staticmethod
    def text_split(documents: UnstructuredFileLoader):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        return texts

    def embeddings(self, texts: list[Document]):
        embeddings = OpenAIEmbeddings()
        self.vectordb = Chroma.from_texts([t.page_content for t in texts], embeddings)
        return self.vectordb

@app.route("/")
def welcome():
    return render_template("welcome.html")

@app.route("/chat")
def chat():
    return render_template("chat.html")

genie = Genie("finished.pdf")
# Genie class and other functions here.
# Handle user queries from the chat interface
@app.route("/ask", methods=["POST"])
def ask_question():
    query = request.json["query"]

    translator = Translator()
    language = translator.detect(query)
    if language.lang == 'hi':
        query = translator.translate(query,dest='en')
        docs = genie.vectordb.similarity_search_with_relevance_scores(query.text)
        docs1 = genie.vectordb.similarity_search(query.text)
        if docs[0][-1] > 1:
            result = genie.fallback(query.text)
            result = translator.translate(result, dest='hi')
            return jsonify({"answer": random.choice(result.text)})
        else:
            result = genie.genie.run(input_documents=docs1, question=query.text)
            result = translator.translate(result, dest='hi')
            return jsonify({"answer": result.text})
    else:
        docs = genie.vectordb.similarity_search_with_relevance_scores(query)
        docs1 = genie.vectordb.similarity_search(query)
        if docs[0][-1] > 1:
            result = genie.fallback(query)
            return jsonify({"answer": random.choice(result)})
        else:
            result =genie.genie.run(input_documents=docs1, question=query)
            return jsonify({"answer": result})


if __name__ == "__main__":
    app.run(debug=True)


