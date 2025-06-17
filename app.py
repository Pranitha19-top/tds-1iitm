from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS (allow requests from anywhere)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Virtual TA API!"}

@app.post("/api/")
async def api_ask_question(q: Question):
    return {
        "answer": f"Thank you for your question: '{q.question}'",
        "links": [
            {
                "url": "https://tds-discourse.iitm.ac.in",
                "text": "TDS Discourse"
            }
        ]
    }
