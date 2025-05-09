def get_model_tokenizer(model_id:str = "google/gemma-3-1b-it"):
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM
    )
    from trl import setup_chat_format
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    from accelerate import PartialState
    device_string = PartialState().process_index
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation='eager',
        device_map={'':device_string}
    )

    if tokenizer.chat_template is None:
        return setup_chat_format(model, tokenizer)
    else:
        return model, tokenizer
