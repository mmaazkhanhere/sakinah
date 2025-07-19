from fastapi import FastAPI

from src.sakinah_agent.agent import agent
from src.sakinah_agent.agent_schema import AgentState

app: FastAPI = FastAPI()

state: AgentState = {
    "user_message": "",
    "chat_history": [],
    "context": [],
    "answer": ""
}

@app.get('/')
def root_message():
    return {"message": "Sakinah Backend System running"}

@app.post('/query')
async def user_query(query: str):
    state["user_message"] = query
    response = agent(state)
    return response