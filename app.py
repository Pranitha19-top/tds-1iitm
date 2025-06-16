from fastapi import FastAPI
from pydantic import BaseModel
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI API Key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"  # Replace with real key or use os.getenv()

class Question(BaseModel):
    question: str
    image: str = None  # Optional field for future image support

@app.get("/")
def home():
    return {"message": "Welcome to Virtual TA API"}

@app.post("/api/")
async def ask_question(q: Question):
    # Load vector DB
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local("vectorstore", embeddings)

    # Similar document search
    docs = db.similarity_search(q.question)

    # Answer generation
    llm = OpenAI(temperature=0)
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=q.question)

    return {
        "answer": answer,
        "links": [  # Add real links if available
            {"url": "https://discourse.iitm.ac.in", "text": "TDS Discourse"}
        ]
    }

