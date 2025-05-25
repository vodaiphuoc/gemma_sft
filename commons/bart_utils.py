import torch
import torch.nn as nn
# from transformers.models.bart.modeling_bart import BartLearnedPositionalEmbedding

class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0, position_ids: torch.Tensor = None):
        """`input_ids' shape is expected to be [bsz x seqlen]."""
        print('position_ids input: ', type(position_ids), position_ids)
        if position_ids is None:
            bsz, seq_len = input_ids.shape[:2]
            position_ids = torch.arange(
                past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
            ).expand(bsz, -1)
        else:
            position_ids = position_ids.unsqueeze(0)

        print('position_ids input: ', type(position_ids), position_ids)
        return super().forward(position_ids + self.offset)



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
    new_decoder_pos_embedding = BartLearnedPositionalEmbedding(new_context_length, embedding_dim)

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