from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import os

load_dotenv()

search_tool = TavilySearch(max_results=3)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def chat_node(state):
    reply = llm_with_tools.invoke(state["messages"])
    return {"messages": [reply]}

def should_use_tool(state):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "end"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    convert_system_message_to_human=True
)

@tool
def search_documents(query: str) -> str:
    """Search the user's uploaded PDF knowledge base for relevant information.
    Use this for questions about documents the user has provided, NOT for live web info."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    vectorstore = Chroma(
        persist_directory="/Users/adityasethi014gmail.com/Desktop/pdf-chat/chroma_db",
        embedding_function=embeddings,
    )
    results = vectorstore.similarity_search(query, k=3)
    return "\n\n".join(doc.page_content for doc in results)

llm_with_tools = llm.bind_tools([search_tool, search_documents])
tool_node = ToolNode([search_tool, search_documents])



graph = StateGraph(AgentState)
graph.add_node("chatter", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chatter")
graph.add_conditional_edges("chatter", should_use_tool, {"tools": "tools", "end": END})
graph.add_edge("tools", "chatter")
app = graph.compile()

result = app.invoke({"messages": [HumanMessage(content="Summarize what the uploaded document is about")]})
content = result["messages"][-1].content
if isinstance(content, list):
    print("".join(part.get("text", "") for part in content if isinstance(part, dict)))
else:
    print(content)