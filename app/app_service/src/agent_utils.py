from typing import Callable, List
import inspect
import json
from pydantic.dataclasses import dataclass
from pydantic.json import pydantic_encoder


########## making of few-shot examples
EXAMPLE1_OUT = json.dumps(
        obj= SingleFinalResponse(
            Thought= 'I need to search information of Hoi An city', 
            Action = Action_Type(
                function_name= 'search_facts', 
                args = [Arg(key='query_search', value = 'where is Hoi An')]
            )
        ), 
        default=pydantic_encoder,
        indent= 4)

EXAMPLE2_OUT = json.dumps(
        obj= SingleFinalResponse(
            Thought= 'I need to find travelling blogs about Viet Nam', 
            Action = Action_Type(
                function_name= 'find_blogs', 
                args = [
                    Arg(key='query_search', value = 'popular location in Viet Name'),
                    Arg(key='k', value = '20')
                ]
            )
        ), 
        default=pydantic_encoder,
        indent= 4)

EXAMPLE3_OUT = json.dumps(
        obj= SingleFinalResponse(
            Thought= 'This question i can answer directly', 
            Action = Action_Type(
                function_name= 'give_direct_answer', 
                args = [
                    Arg(key="direct_answer", 
                        value = """Vietnam is located in Southeast Asia. It is situated on the eastern edge of 
                the Indochina Peninsula, bordering China to the north, Laos and Cambodia to the west, and the 
                South China Sea to the east"""
                    )
                ]
            )
        ), 
        default=pydantic_encoder,
        indent= 4)

