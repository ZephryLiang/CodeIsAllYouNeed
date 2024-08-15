from transformers import AutoTokenizer
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
import os
local_dir = os.path.expanduser('~')
distilbert_dir =  f"{local_dir}/.{model_ckpt}"
if not os.path.exists(distilbert_dir):
    os.mkdir(path=distilbert_dir)
print(f"local_dir:{local_dir}")
try:
    tokenizer.save(local_dir+'/'+model_ckpt)
except Exception as e:
    print(f"tokenizer.save error happen :{str(e)}")
from transformers import DistilBertTokenizer 
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(model_ckpt,tokenize_chinese_chars=True,pad_token='[pad]',unk_token="[UNK]")
try:
    distilbert_tokenizer.save(local_dir+'/'+model_ckpt)
except Exception as e:
    print(f"distilbert_tokenizer.save error happen :{str(e)}")

try:
    distilbert_tokenizer.save_pretrained(distilbert_dir)
    print(f"distilbert_tokenizer.save_pretrained succeed at :{str(distilbert_dir)}") 
except Exception as e:
    print(f"distilbert_tokenizer.save_pretrained error happen :{str(e)}")    
text = "Tokenizing text is a core task of NLP.测试下中文切分" 
#encoded_text = distilbert_tokenizer.encode_batch(text,pair = None,is_pretokenized = False,add_special_tokens = True)
encoded_text = distilbert_tokenizer.tokenize(text
                                             )#source code __init__ 提到了self.basic_tokenizer
print(f"distilbert-tokenizer property sep_token value=={distilbert_tokenizer.sep_token}")
print(f"distilbert_tokenizer encoded_text :{str(encoded_text)}")
print('how to translate vocab.txt to human-reable,currently it is in a numerical format')



print(f"distilbert_tokenizer.special_tokens_map :{distilbert_tokenizer.special_tokens_map}")
print(f"distilbert_tokenizer.all_special_tokens :{distilbert_tokenizer.all_special_tokens}")
print(f"distilbert_tokenizer.cls_token_id :{distilbert_tokenizer.cls_token_id}")
encoded_plus = distilbert_tokenizer.encode_plus(text
                                             )#source code __init__ 提到了self.basic_tokenizer

print(f"distilbert_tokenizer encoded_plus :{str(encoded_plus)}")
#print(f"distilbert_tokenizer self.ids_to_tokens :{(distilbert_tokenizer.ids_to_tokens)}")
#print("以上实际上是vocab_index，后续对于任何输入都通过这个Index去做标记映射")

print(f"distilbert_tokenizer id2token :{distilbert_tokenizer.convert_ids_to_tokens(encoded_plus['input_ids'])}")
#thinking about the index mechanism how to achieve between id and token
print(f"distilbert_tokenizer token2string :{distilbert_tokenizer.convert_tokens_to_string(encoded_text)}")


def tokenize_batch(batch):
    input_ids = distilbert_tokenizer.encode(  text=batch[0],text_pair=batch[1])
    print(input_ids)
    tokens  = distilbert_tokenizer.convert_ids_to_tokens(input_ids)
    return tokens
print(f"test text_pair usage:{tokenize_batch(['youarehandsome','tokkenizepipelinehas 4 steps'])}")