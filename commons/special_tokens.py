
class GemmaLikeSpecialTokens:

    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    pad_token: str = "<pad>"

def adjust_tokenizer(model, tokenizer):

    tokenizer.eos_token = GemmaLikeSpecialTokens.eos_token
    tokenizer.pad_token = GemmaLikeSpecialTokens.pad_token
    tokenizer.bos_token = GemmaLikeSpecialTokens.bos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [GemmaLikeSpecialTokens.bos_token, GemmaLikeSpecialTokens.eos_token]})
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    
    # Update the generation config to use the new eos & bos token
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    num_added_tokens = tokenizer.add_tokens(["<start_of_turn>", "<end_of_turn>"])
    print("added", num_added_tokens, "tokens")
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer