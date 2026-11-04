from transformers import AutoTokenizer

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B")

tokens = tokenizer("你好，世界！", add_special_tokens=False, return_tensors="pt")

print(tokens)

text = "你好，世界！<|endoftext|>"
tokens = tokenizer(text, return_tensors="pt")
print(tokens)

# 查看特殊 Token
print("特殊 Token:", tokenizer.special_tokens_map)

# 查看具体的 Token ID
print("特殊 Token ID:", {tok: tokenizer.convert_tokens_to_ids(tok) for tok in tokenizer.special_tokens_map.values()})
