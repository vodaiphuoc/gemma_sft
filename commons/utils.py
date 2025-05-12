import yaml
from typing import Dict, Union
from types import NoneType

def get_fsdp_config_from_yaml(yaml_path: str)->Union[Dict[str,str], NoneType]:
    if yaml_path != "":
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.load(f, Loader=yaml.SafeLoader)
            return data['fsdp_config']
        
        except Exception as e:
            print("error in open file yaml config: ",e)
        
    else:
        return None
