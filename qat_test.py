from commons.model import get_model_tokenizer
import torch
from transformers import CompileConfig, HybridCache, GenerationConfig

precompile_model, tokenizer, _, _ = get_model_tokenizer(
    model_key= "gemma",
    distribution_device= "cuda", 
    distribution_type = "ddp",
    checkpoint_dir = "checkpoints/2025-05-29_11-44-59"
)


_generation_config = {
    "max_new_tokens": 64,
    "do_sample": True,
    "temperature": 1.0,
    "top_k": 64,
    "top_p": 0.95,
    # "compile_config": CompileConfig(
    #     backend = "inductor",
    #     mode="default",
    # )
}


model = precompile_model.cpu()
# model = torch.compile(
#     model, 
#     mode="default",
#     backend="inductor"
# )
model.eval()

print('model dtype: ',model.dtype)

past_key_values = HybridCache(
    config=model.config, 
    max_batch_size=1, 
    max_cache_len= 60 + _generation_config['max_new_tokens'], 
    device=model.device, 
    dtype=torch.int8
)

inputs = tokenizer(
    "feel sad today",
    add_special_tokens = False,
    padding = "max_length",
    truncation= True,
    max_length= 60,
    padding_side = 'left',
    return_tensors = 'pt'
)

res = model.generate(
    **inputs, 
    # past_key_values=past_key_values, 
    generation_config = GenerationConfig(**_generation_config),
    use_cache=False,
    use_model_defaults = False,
)


print(res)