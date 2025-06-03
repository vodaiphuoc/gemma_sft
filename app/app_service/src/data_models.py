from pydantic.dataclasses import dataclass
from pydantic import computed_field
from typing import List, Dict, Any


########### for Chroma DB
@dataclass
class DocContext:
    ith:int
    content:str
    sim_score:float

    @computed_field
    @property
    def to_string(self)->str:
        return f"""
{{
    Content: {self.content}
    Relevance score: {self.sim_score}
}}
"""

@dataclass
class Context:
    query_seach: str
    contexts: List[DocContext]

    @computed_field
    @property
    def to_string(self)->str:
        results_str = "".join([_res.to_string for _res in self.contexts])

        return f"""
search blogs results:
{results_str}
"""    

