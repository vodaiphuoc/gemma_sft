import torch.nn as nn
import torch.nn.functional as F
import torch
from dataclasses import dataclass
from typing import Optional

from transformers import (
    GenerationMixin,
    PreTrainedModel,
    AutoModelForCausalLM
)

from transformers.models.gemma3.modeling_gemma3 import Gemma3TextScaledWordEmbedding
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class LSTMConfig(PretrainedConfig):
    model_type = "CustomLSTM"
    def __init__(
            self,
            pad_token_id:int = 0,
            eos_token_id:int = 1,
            bos_token_id:int = 2,
            tie_word_embeddings:bool = True,
            initializer_range:float = 0.02,
            sequence_length: int = 2048,
            vocab_size: int = 262208,
            embedding_dim: int = 512,
            hidden_size: int = 512,
            num_lstm_layer: int = 4,
            dropout: float = 0.1,
            bias: bool = True
        )->None:
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
        )
        self.initializer_range = initializer_range
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_lstm_layer = num_lstm_layer
        self.dropout = dropout
        self.bias = bias

class LSTMTextModel(nn.Module):
    def __init__(self, config: LSTMConfig):
        super().__init__()
        
        self.embedding = Gemma3TextScaledWordEmbedding(
            num_embeddings = config.vocab_size, 
            embedding_dim = config.embedding_dim,
            padding_idx= config.pad_token_id
        )
        self.lstm = nn.LSTM(
            input_size = config.embedding_dim, 
            hidden_size = config.hidden_size, 
            num_layers = config.num_lstm_layer,
            bias = True,
            batch_first = True,
            bidirectional = False,
            proj_size = 0
            )
        self.fc = nn.Linear(config.hidden_size, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
            self, 
            input_ids: torch.LongTensor = None
        )->torch.Tensor:
        r"""
        Args:
            input_ids (torch.LongTensor)
        Returns:
            torch.Tensor shape (N, L, config.vocab_size)
        """
        embeds = self.embedding(input_ids)
        
        # output lstm shape: (N, L, config.hidden_size)
        output, _ = self.lstm(embeds)

        output = self.dropout(output)
        return self.fc(output)

    # def generate(self, idx, max_new_tokens):
    #     for _ in range(max_new_tokens):
    #         idx_cond = idx[:, -config['block_size']:]
    #         embeds = self.embedding(idx_cond)
    #         output, _ = self.rnn(embeds)
    #         logits = self.fc(output[:, -1, :])
    #         probs = F.softmax(logits, dim=-1)
    #         idx_next = torch.multinomial(probs, num_samples=1)
    #         idx = torch.cat((idx, idx_next), dim=1)
    #     return idx
    
class CustomLSTMForCausalLM(PreTrainedModel):
    config_class = LSTMConfig

    def _init_weights(self, module):
        print('call init weight', type(module))
        std = self.config.initializer_range

        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, Gemma3TextScaledWordEmbedding):
            module.weight = AutoModelForCausalLM.from_pretrained(
                "google/gemma-3-1b-it",
                attn_implementation='eager',
                torch_dtype=torch.float32,
            ).get_input_embeddings.weight

            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        
    def __init__(self, config: LSTMConfig):
        super().__init__(config)
        self.model = LSTMTextModel(config = config)
        self.post_init()

    def forward(
            self, 
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None, 
            labels: Optional[torch.LongTensor] = None
        )->CausalLMOutputWithPast:
        r"""
        Args:
            input_ids (torch.LongTensor)
            labels (torch.LongTensor): shape (N, L)
        """
        logits = self.model(input_ids = input_ids)
        loss = self.loss_function(logits, labels, self.config.vocab_size)

        return CausalLMOutputWithPast(loss = loss, logits= logits)