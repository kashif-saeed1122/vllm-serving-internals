https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ/blob/main/config.json

Usefull info 


Field	Expected	Why it matters
num_hidden_layers	28	multiplier in the KV formula
num_attention_heads	28	NOT the one you want
num_key_value_heads	4	GQA. Using 28 here overestimates KV by 7×
hidden_size	3584	→ head_dim = 3584 / 28 = 128
tie_word_embeddings	false	embed + lm_head are separate → ~2.2 GiB of fp16 weights paid twice
