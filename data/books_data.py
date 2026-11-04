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
        self.split = split
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B")
        dataset = load_dataset("yuyijiong/Book_Summary_Chinese", split=split, cache_dir="./data/books")
        text = "".join(data['chapter'] for data in dataset)

        tokenized_output = self.tokenizer(
            text,
            add_special_tokens=False,  # 不添加特殊 token
            return_tensors="pt",  # 返回 PyTorch Tensor
            truncation=True,  # **确保 token 长度不会超出 max_length**
            max_length=131072  # **防止 token 过长**
        )
        self.tokens = tokenized_output['input_ids'].view(-1).to(self.device)
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
