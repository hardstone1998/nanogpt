from datasets import load_dataset

split_names = load_dataset("yuyijiong/Book_Summary_Chinese", cache_dir="../data/books")
print(split_names)