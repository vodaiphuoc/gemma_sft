def get_model_tokenizer(model_id:str = "google/gemma-3-1b-it"):
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    from accelerate import PartialState
    device_string = PartialState().process_index
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation='eager',
        device_map={'':device_string}
    )
    return model, tokenizer
