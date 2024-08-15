import collections
import csv

# vocab.txt 文件路径
vocab_file = "/home/desir/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084"

with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
        print(' '.join(str(token) for token in tokens))
# 将词汇写入一个可读的 CSV 文件
output_file = "human_readable_vocab.csv"
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Index", "Word"])
    
    for index, token in enumerate(tokens):
        token = token.rstrip('\n')
        writer.writerow([index, token])

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    print(f"----------------------------------------------vocab file path:{vocab_file}")
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip('\n')
        vocab[token] = index
    #print(f"vocab file load ends:{vocab}")
    return vocab
print(f"词汇表已保存为 {output_file}")
