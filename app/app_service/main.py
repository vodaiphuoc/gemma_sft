from src.agent import Agent, ReponseStatus

from fastapi import FastAPI, Request, Response, Body, status, Depends, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder

from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from typing import List, Any, Annotated, Literal, Union, Dict
from typing import Annotated

import json
import re
from contextlib import asynccontextmanager
from loguru import logger
import asyncio
import uvicorn

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    def get_active_connections(self):
        print(self.active_connections)
        print('num active connection: ', len(self.active_connections))

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str,websocket: WebSocket, topic = None):
        if topic is None:
            await websocket.send_text(json.dumps({'msg': message}))
        else:
            await websocket.send_text(json.dumps({'msg': message, 'topic': topic}))

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Setting up System")
    app.manager = ConnectionManager()
    app.agent = Agent()

    yield
    logger.info("Turning down system")
    app.agent.chat_hist_db.close()
    app.agent = None
    app.manager = None

app = FastAPI(lifespan = lifespan)
app.mount(path = '/templates', 
          app = StaticFiles(directory='front_end/templates', html = True), 
          name='templates')

app.mount(path = '/static',
          app = StaticFiles(directory='front_end/static', html = False), 
          name='static')

origins = [
    "http://localhost",
    "http://localhost:8080/",
    "http://localhost:8000/",
    "http://127.0.0.1:8000/",
    "http://127.0.0.1:8000"
]
app.add_middleware(CORSMiddleware, 
                   allow_origins=origins,
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"]
                   )
templates = Jinja2Templates(directory='front_end/templates')

@app.get("/", response_class=HTMLResponse)
async def index_router(request: Request):
    return templates.TemplateResponse(
		request = request,
		name = "index.html"
		)

@app.post("/load_topics", response_class=JSONResponse)
async def load_history(request: Request):
    """
    Load all available topics in DB
    Return type:
        List[str]: List of topic names
    """
    all_topics = request.app.agent.chat_hist_db.get_topics()

    return JSONResponse(
        status_code = 200, 
        content = all_topics
    )


@app.post("/load_history", response_class=JSONResponse)
async def load_history(
    body_data: Annotated[Dict[str,str], Body()],
    request: Request):
    """
    Given a topic value, query in database to get chat history
    Return type:
        List[Dict[str,str]] where a dictionary has keys 'role' and 'msg'
    """
    topic_value = body_data['topic_value']
    chat_history = request.app.agent.chat_hist_db.get_chat_history(topic_value)

    return JSONResponse(
        status_code = 200, 
        content = chat_history
    )


@app.websocket("/ws")
async def chat_router(websocket: WebSocket):
    await websocket.app.manager.connect(websocket)
    
    try:
        while True:
            data = await websocket.receive_json()
            websocket.app.manager.get_active_connections()
            # unpack data
            use_timestamp = data['time_stamp']
            user_message = data['user_message']
            topic_value = user_message[:35] if data['topic'] == '' else data['topic']

            # agent inference
            logger.debug('user_message: {}'.format(user_message))
            agent_response: ReponseStatus = websocket.app.agent(
                prompt_data = user_message, 
                topic =  None if data['topic'] == '' else data['topic']
            )

            # add to history chat of current topic
            if agent_response.status_code == 200:
                websocket.app.agent.chat_hist_db.insert_new_turns(
                    topic = topic_value,
                    new_msgs = [
                        {"role": "user", "content": user_message, 'timestamp': use_timestamp},
                        {"role": "model", "content": agent_response.answer_response, 'timestamp': use_timestamp}
                    ]
                )
            
            if agent_response.status_code == 200:
            # send data
                await websocket.app.manager.send_personal_message(
                    message = agent_response.answer_response,
                    topic = topic_value if data['topic'] == '' else None,
                    websocket= websocket)
            else:
                await websocket.app.manager.send_personal_message(
                    message = agent_response.toStrFailed,
                    topic = topic_value if data['topic'] == '' else None,
                    websocket= websocket)

    except WebSocketDisconnect:
        websocket.app.manager.disconnect(websocket)


async def main_run():
    config = uvicorn.Config("main:app", 
    	port=8080, 
    	reload=True,
		reload_dirs= ["front_end/static", "front_end/templates"]
    	)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main_run())