
class GemmaLikeTemplate:

    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    pad_token: str = "<pad>"

    @property
    def chat_template(self):
        return "{{ bos_token }}\n{%- if messages[0]['role'] == 'system' -%}\n    {%- if messages[0]['content'] is string -%}\n        {%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}\n    {%- else -%}\n        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}\n    {%- endif -%}\n    {%- set loop_messages = messages[1:] -%}\n{%- else -%}\n    {%- set first_user_prefix = \"\" -%}\n    {%- set loop_messages = messages -%}\n{%- endif -%}\n{%- for message in loop_messages -%}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}\n        {{ raise_exception(\"Conversation roles must alternate user/assistant/user/assistant/...\") }}\n    {%- endif -%}\n    {%- if (message['role'] == 'assistant') -%}\n        {%- set role = \"model\" -%}\n    {%- else -%}\n        {%- set role = message['role'] -%}\n    {%- endif -%}\n    {{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else \"\") }}\n    {%- if message['content'] is string -%}\n        {{ message['content'] | trim }}\n    {%- elif message['content'] is iterable -%}\n        {%- for item in message['content'] -%}\n            {%- if item['type'] == 'image' -%}\n                {{ '<start_of_image>' }}\n            {%- elif item['type'] == 'text' -%}\n                {{ item['text'] | trim }}\n            {%- endif -%}\n        {%- endfor -%}\n    {%- else -%}\n        {{ raise_exception(\"Invalid content type\") }}\n    {%- endif -%}\n    {{ '<end_of_turn>\n' }}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{'<start_of_turn>model\n'}}\n{%- endif -%}\n",


def adjust_tokenizer(model, tokenizer):

    tokenizer.eos_token = GemmaLikeTemplate.eos_token
    tokenizer.pad_token = GemmaLikeTemplate.pad_token
    tokenizer.bos_token = GemmaLikeTemplate.bos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [GemmaLikeTemplate.bos_token, GemmaLikeTemplate.eos_token]})
    # set chat format for tokenizer
    tokenizer.chat_template = GemmaLikeTemplate.chat_template

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    # Update the generation config to use the new eos & bos token

    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer