import torch
from datasets import load_dataset
from transformers import AutoTokenizer


class BooksDataLoaderLite:
    def __init__(self, B, T, device, process_rank, num_processes, split="train"):
        self.B = B
        self.T = T
        self.device = device
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B")
        if split == "train":
            self.chunk_size = 80000
        else:
            self.chunk_size = 20000
        dataset = (load_dataset("qgyd2021/chinese_porn_novel", cache_dir="./data/novel")["train"]
                   .train_test_split(test_size=0.2, seed=42)[split])
        self.text = "".join(data['content'] for data in dataset)

        self.reset()
        print(f"加载 {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + (B * T + 1)]
        x = (buf[:-1]).view(B, T).to(self.device)
        y = (buf[1:]).view(B, T).to(self.device)

        self.current_position += B * T * self.num_processes

        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x, y

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = self.load_tokens(self.text)

    def load_tokens(self, text):
        tokens = []

        # 按照 chunk_size 分批处理文本，避免一次性加载过多数据
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i+self.chunk_size]
            tokenized_chunk = self.tokenizer(chunk,
                                             add_special_tokens=False,  # 不添加特殊 token
                                             return_tensors="pt",  # 返回 PyTorch Tensor
                                             truncation=True,  # **确保 token 长度不会超出 max_length**
                                             max_length=131072  # **防止 token 过长**
                                            )
            tokens.append(tokenized_chunk['input_ids'].view(-1))

        # 将分批的 tokens 合并
        return torch.cat(tokens, dim=0)