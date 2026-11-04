

from datasets import load_dataset

dataset1 = load_dataset("yuyijiong/Book_Summary_Chinese",cache_dir="./books")
# dataset2 = load_dataset("qgyd2021/chinese_porn_novel",cache_dir="./novel")
print(dataset1)
# print(dataset2)

train_data1 = dataset1["train"]
# train_data2 = dataset2["train"]
print(train_data1[0])
# print(train_data2[0])
for data in train_data1:
    print(data['chapter'])