from typing import Union, List, Literal, Dict, Optional
import os
import json
from pydantic.dataclasses import dataclass
from pydantic import computed_field
from collections.abc import Iterable
import requests
from src.structuredb import ChatHistoryDB
import copy
from dotenv import load_dotenv
from loguru import logger

@dataclass
class Answer:
    predict_emotion: str
    response:str

@dataclass
class ReponseStatus:
    status_msg: Literal["success","backend failed"]
    status_code: int
    answer_response: Union[None, str]

    @computed_field
    @property
    def toStrFailed(self)->str:
        return f"status code: {self.status_code}, msg: {self.status_msg}"

class Agent(object):
    SYSTEM_PROMT = "You are a helpfull assistant always give emotional reponse in conservation.\nYour reponse must follow the JSON format:\n{\n    \"predict_emotion\": str\n    \"response\": str\n}\nwhere \"predict_emotion\" can be one of following labels:\n    'sentimental', 'impressed', 'proud', 'devastated', 'content', 'afraid', 'surprised', \n    'hopeful', 'prepared', 'furious', 'faithful', 'angry', 'annoyed', 'sad', 'embarrassed', \n    'confident', 'ashamed', 'apprehensive', 'terrified', 'disappointed', 'lonely', 'jealous', \n    'anxious', 'grateful', 'caring', 'guilty', 'disgusted', 'excited', 'nostalgic', 'joyful', \n    'anticipating', 'trusting'.\nand \"response\" is your main emotional response.\n"

    _generation_config = {
        "model": "ftlora_main",
        "max_completion_tokens": 120,
        "n": 1,
        "seed": 200,
        "stream": False,
        "temperature": 1.0,
        "top_p": 0.95,
        "use_beam_search": False,
        "top_k": 64,
        "min_p": 0,
        "repetition_penalty": 1.0,
        "length_penalty": 1,
        "stop_token_ids": [
            1, 106
        ],
        "include_stop_str_in_output": False,
        "ignore_eos": True,
        "min_tokens": 10,
        "skip_special_tokens": True,
        "spaces_between_special_tokens": True,
        "echo": False,
        "add_generation_prompt": True,
        "continue_final_message": False,
        "add_special_tokens": True,
        "return_tokens_as_token_ids": False
    }

    def __init__(self):
        super().__init__()
        load_dotenv()
        self._inference_domain = os.environ['inference_service_domain']
        
        self.chat_hist_db = ChatHistoryDB()
        logger.info("done init Agent")
    
    def __call__(self, prompt_data:str, topic:str = None)->ReponseStatus:
        if topic is not None:
            chat_history = [
                {
                    'role': chat['role'],
                    'content': chat['content']
                }
                for chat in self.chat_hist_db.get_chat_history(topic= topic)
            ]
        else:
            chat_history = []

        # construct messages
        messages = [{
            "content": self.SYSTEM_PROMT,
            "role": "system"
        }]
        if len(chat_history) != 0:
            messages.extend(chat_history)

        messages.append({
            "content": prompt_data,
            "role": "user"
        })

        send_body = copy.deepcopy(self._generation_config)
        send_body['messages'] = messages

        responses = requests.post(
            url = self._inference_domain + "/v1/chat/completions",
            data = json.dumps(send_body),
            headers = {"Content-Type":"application/json"}
        )
        if responses.ok:
            data = responses.json()['choices'][0]['message']['content']

            data_dict = None
            try:
                data_dict = json.loads(data)
            except Exception as e:
                logger.error(f"reponse not in dict type, {e}")
                return ReponseStatus(
                    status_msg= "backend failed", 
                    status_code= 500, 
                    answer_response=None
                )
            
            try:
                assert data_dict is not None
                ans = Answer(**data_dict)
                logger.info(f"prediction emotion, {ans.predict_emotion}")
                return ReponseStatus(
                    status_msg= "success", 
                    status_code= 200, 
                    answer_response=ans.response
                )

            except Exception as e:
                logger.error(f"cannot parse to Answer, {e}")
                return ReponseStatus(
                    status_msg= "backend failed", 
                    status_code= 500,
                    answer_response=None
                )
            
        else:
            return ReponseStatus(
                status_msg= "backend failed", 
                status_code= 500, 
                answer_response="hi"
            )
