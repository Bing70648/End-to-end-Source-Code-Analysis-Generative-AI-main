from src.helper import repo_ingestion, load_repo, text_splitter, load_embedding
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
import os

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY


# url = "https://github.com/Bing70648/End-to-end-Source-Code-Analysis-Generative-AI-main"

# repo_ingestion(url)


documents = load_repo("repo/")
text_chunks = text_splitter(documents)
embeddings = load_embedding()


# storing vector in choramdb
vectordb = Chroma.from_documents(
    text_chunks, embedding=embeddings, persist_directory="./db"
)
vectordb.persist()
