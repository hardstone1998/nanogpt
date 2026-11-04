import os

from data.novel_data import BooksDataLoaderLite

total_batch_size = 131072  # 2**19, ~0.5M, in number of tokens
B = 4  # micro batch size
T = 512  # sequence length
device = "cuda"
ddp_rank = 0
ddp_world_size = 1
train_loader = BooksDataLoaderLite(B=B, T=T, device=device, process_rank=ddp_rank, num_processes=ddp_world_size,
                                   split="train")
x, y = train_loader.next_batch()
print(x.shape)
print(y.shape)