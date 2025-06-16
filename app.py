from fastapi import FastAPI
from pydantic import BaseModel
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for local frontend development (optional but useful)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set your OpenAI API key here or use an environment variable
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"  # Replace with your actual API key or use env variable

# Request body format
class Question(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the PDF Q&A API!"}

@app.post("/ask")
async def ask_question(q: Question):
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local("vectorstore", embeddings)
    docs = db.similarity_search(q.question)
    llm = OpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=q.question)
    return {"answer": answer}

@app.post("/api/")
async def api_ask_question(q: Question):
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local("vectorstore", embeddings)
    docs = db.similarity_search(q.question)
    llm = OpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=q.question)
    return {"answer": answer}

