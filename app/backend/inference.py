# from transformers import BitsAndBytesConfig
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
# )
# from peft import PeftModel
# import torch


# class Serving(object):
#     _generation_config = {
#         "max_new_tokens": 64,
#         "do_sample": True,
#         "temperature": 1.0,
#         "top_k": 64,
#         "top_p": 0.95
#     }

#     _max_input_length = 128

#     def __init__(
#         self,
#         checkpoint_dir:str
#         ):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#         quantization_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_compute_dtype=torch.bfloat16,
#             bnb_4bit_quant_storage=torch.bfloat16
#         )
        
#         base_model = AutoModelForCausalLM.from_pretrained(
#             "google/gemma-3-1b-it",
#             attn_implementation='eager',
#             torch_dtype="auto",
#             quantization_config = quantization_config,
#         )

#         model = PeftModel.from_pretrained(
#             base_model, 
#             checkpoint_dir
#         )
#         model.eval()
#         self.model = model.to(self.device)

#         self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

#     def inference(self, input_question:str):
#         inputs = self.tokenizer.apply_chat_template(
#             input_question,
#             tokenize= True, 
#             add_generation_prompt=True,
#             padding = "max_length",
#             truncation= True,
#             max_length= self._max_input_length,
#             padding_side = 'left',
#             return_tensors = 'pt'
#         ).to(self.device)
            
#         with torch.inference_mode():
#             outputs = self.model.generate(
#                 **inputs, 
#                 **self._generation_config
#             )
        
#         return {
#             "model_response": self.tokenizer.batch_decode(
#                 outputs[:, inputs['input_ids'].shape[1]:], 
#                 skip_special_tokens = True
#                 )
#         }



# from vllm import LLM, SamplingParams
# from vllm.lora.request import LoRARequest
# import torch
# import os

# class Serving(object):
#     _lora_path = 
#     def __init__(self):
#         self.llm = LLM(
#             model="google/gemma-3-1b-it", 
#             dtype=torch.bfloat16,
#             enable_lora=True,
#             trust_remote_code=True,
#             quantization="bitsandbytes", 
#             load_format="bitsandbytes"
#         )

#         self.sampling_params = SamplingParams(
#             temperature = 1.0,
#             max_tokens = 64,
#             top_k = 64,
#             top_p = 0.95,
#             stop=["<eos>"],
#             ignore_eos = False,
#         )

#     def inference(self, messages):
#         outputs = self.llm.generate(
#             messages,
#             self.sampling_params,
#             lora_request = LoRARequest(
#                 lora_name= "adapter",
#                 lora_int_id= 1, 
#                 lora_local_path = self._lora_path 
#                 )
#         )