from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
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
llm_with_tools = llm.bind_tools([search_tool])
tool_node = ToolNode([search_tool])

graph = StateGraph(AgentState)
graph.add_node("chatter", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chatter")
graph.add_conditional_edges("chatter", should_use_tool, {"tools": "tools", "end": END})
graph.add_edge("tools", "chatter")
app = graph.compile()

result = app.invoke({"messages": [HumanMessage(content="What are the latest Julian Alvarez transfer rumors in 2025?")]})
content = result["messages"][-1].content
if isinstance(content, list):
    print("".join(part.get("text", "") for part in content if isinstance(part, dict)))
else:
    print(content)