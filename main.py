from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from research_agent import app as agent

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    
@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
   result = agent.invoke({"messages": [HumanMessage(content=request.question)]})    
   content = result["messages"][-1].content
   return QueryResponse(answer=str(content))

