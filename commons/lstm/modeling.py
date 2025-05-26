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

from typing import Tuple

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
            vocab_size: int = 262144,
            embedding_dim: int = 1152,
            hidden_size: int = 256,
            num_lstm_layer: int = 4,
            bidirectional:bool = False,
            dropout: float = 0.1,
            bias: bool = True,
            num_lsmt_block: int = 4
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
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.bias = bias
        self.num_lsmt_block = num_lsmt_block

class LSTMBlock(nn.Module):
    def __init__(self, config: LSTMConfig, input_size:int, output_size:int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = config.hidden_size, 
            num_layers = config.num_lstm_layer,
            bias = True,
            batch_first = True,
            bidirectional = False,
            )
        
        self.project_back1 = nn.Linear(config.hidden_size, output_size)
        
    def forward(
            self, 
            embeds: torch.Tensor, 
            previous_block_hn: torch.Tensor = None, 
            previous_block_cn: torch.Tensor = None
        )->Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        # output lstm shape: (N, L, config.hidden_size)
        if previous_block_hn is None and previous_block_cn is None:
            output, (hn, cn) = self.lstm(embeds)
        else:    
            output, (hn, cn) = self.lstm(embeds, (previous_block_hn, previous_block_cn))
        return self.project_back1(output), (hn, cn)

class LSTMTextModel(nn.Module):
    def __init__(self, config: LSTMConfig):
        super().__init__()
        
        self.embedding = Gemma3TextScaledWordEmbedding(
            num_embeddings = config.vocab_size, 
            embedding_dim = config.embedding_dim,
            padding_idx= config.pad_token_id
        )

        delta_size = (config.embedding_dim - config.hidden_size)//(config.num_lsmt_block -1)
        size_list = list(reversed([_i 
         if _i + delta_size < config.embedding_dim
         else config.embedding_dim
         for _i in range(
            config.hidden_size, 
            config.embedding_dim, 
            delta_size
        )]
        ))
        assert len(size_list) == config.num_lsmt_block

        self.lstm_blocks = nn.ModuleList(
            [
                LSTMBlock(
                    config, 
                    size_list[_i],
                    size_list[_i+1],
                ) if _i != len(size_list) -1
                else LSTMBlock(
                    config, 
                    size_list[_i],
                    config.hidden_size,
                )

                for _i in range(len(size_list))
            ]
        )
        self.fc = nn.Linear(config.hidden_size, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)
        self.config = config

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
        B = input_ids.shape[0]
        embeds = self.embedding(input_ids)
        
        hn = torch.zeros(
            (2*self.config.num_lstm_layer, B, self.config.hidden_size), 
            dtype= embeds.dtype,
            device= embeds.device
            ) if \
            self.config.bidirectional else \
            torch.zeros(
                (1*self.config.num_lstm_layer, B, self.config.hidden_size),
                dtype= embeds.dtype,
                device= embeds.device
                )
        
        cn = torch.zeros(
            (2*self.config.num_lstm_layer, B, self.config.hidden_size), 
            dtype= embeds.dtype,
            device= embeds.device
            ) if \
            self.config.bidirectional else \
            torch.zeros(
                (1*self.config.num_lstm_layer, B, self.config.hidden_size),
                dtype= embeds.dtype,
                device= embeds.device
                )
        
        for block in self.lstm_blocks:
            embeds, (hn, cn) = block(
                embeds = embeds,
                previous_block_hn = hn, 
                previous_block_cn = cn
            )
        
        output = self.dropout(embeds)
        return self.fc(output)
    
class CustomLSTMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = LSTMConfig

    def _init_weights(self, module):
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
            ).get_input_embeddings().weight
        
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