import torch
import torch.nn as nn

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

def extend_position_embedding(model, new_context_length:int = 2048):
    print('input model type: ',type(model))
    print('input model.model type: ',type(model.model))

    original_context_length = model.config.max_position_embeddings
    embedding_dim = model.config.d_model

    print(f"Original context length: {original_context_length}")
    print(f"Embedding dimension: {embedding_dim}")

    old_decoder_pos_embedding = model.model.decoder.embed_positions
    new_decoder_pos_embedding = nn.Embedding(new_context_length, embedding_dim)

    print(f"Created new positional embedding layers with size ({new_context_length}, {embedding_dim}).")

    # Copy for decoder
    with torch.no_grad():
        # Copy existing weights for positions 0 to original_context_length - 1
        new_decoder_pos_embedding.weight[:original_context_length, :] = old_decoder_pos_embedding.weight[:original_context_length, :]
        print(f"Copied {original_context_length} weights from old decoder PE to new decoder PE.")

    # 5. Assign the new embedding layers back to the model
    model.model.decoder.embed_positions = new_decoder_pos_embedding

    # Update the model's configuration to reflect the new max position embeddings
    model.config.max_position_embeddings = new_context_length
    print(f"Model's max_position_embeddings updated to {model.config.max_position_embeddings}.")

    model.model.decoder.embed_positions.weight.requires_grad = True

    return model