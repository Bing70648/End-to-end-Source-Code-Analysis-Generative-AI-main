from langchain.vectorstores import Chroma
from src.helper import load_embedding
from dotenv import load_dotenv
import os
from src.helper import repo_ingestion
from flask import Flask, render_template, jsonify, request
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain


app = Flask(__name__)


load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY


embeddings = load_embedding()
persist_directory = "db"
# Now we can load the persisted database from disk, and use it as normal.
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)


llm = ChatGoogleGenerativeAI(
    google_api_key=GEMINI_API_KEY,
    model="gemini-2.0-flash",  # 或者 'gemini-2'，取决于您使用的模型版本
    temperature=0.0,
)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
)


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")


@app.route("/chatbot", methods=["GET", "POST"])
def gitRepo():

    if request.method == "POST":
        user_input = request.form["question"]
        repo_ingestion(user_input)
        os.system("python store_index.py")

    return jsonify({"response": str(user_input)})


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)

    if input == "clear":
        os.system("rm -rf repo")

    result = qa(input)
    print(result["answer"])
    return str(result["answer"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
