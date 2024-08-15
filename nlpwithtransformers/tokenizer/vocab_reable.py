from transformers import DistilBertTokenizer

def convert_special_tokens_to_readable(tokenizer, output_file):
    # 获取特殊标记
    special_tokens_map = tokenizer.special_tokens_map
    all_special_tokens = tokenizer.all_special_tokens
    
    # 定义标记的描述
    descriptions = {
        '[CLS]': '[CLS] (Classification Token)',
        '[SEP]': '[SEP] (Separator Token)',
        '[PAD]': '[PAD] (Padding Token)',
        '[MASK]': '[MASK] (Mask Token)',
        '[UNKNOWN]': '[UNKNOWN] (Unknown Token)'
    }

    with open(output_file, 'w') as file:
        # 写入特殊标记的描述
        file.write("Special Tokens:\n")
        for token_name, token in special_tokens_map.items():
            description = descriptions.get(token, f'{token} (Unused Token)')
            file.write(f"{token_name}: {description}\n")
        
        # 写入所有特殊标记
        file.write("\nAll Special Tokens:\n")
        for token in all_special_tokens:
            description = descriptions.get(token, f'{token} (Unused Token)')
            file.write(f"{description}\n")
from distbert import distilbert_dir
# 加载 DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(distilbert_dir)

# 使用脚本
convert_special_tokens_to_readable(tokenizer, 'readable_special_tokens.txt')
