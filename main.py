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
   if isinstance(content, list):
    answer = "".join(part.get("text", "") for part in content if isinstance(part, dict))
   else:
            answer = str(content)
   return QueryResponse(answer=answer)

