import torch.nn.functional as F
import pypdf
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-small-v2')

contents = []
# creating a pdf reader object

reader = pypdf.PdfReader(r"C:\Users\prath\Downloads\Mechanics of Materials 10th Edition by Russell C. Hibbeler (z-lib.org) (5).pdf")
for i in tqdm(range(len(reader.pages))):
    content = reader.pages[i].extract_text()
    for j in tokenizer.encode(content, max_length=505, padding=True,truncation=True, return_overflowing_tokens=True):
        tok = j
        if len(j)<10:
            continue
        prefix = tokenizer.encode('passage: ')[:-1]
        prefix.extend(tok[1:])
        contents.append(prefix)



def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

question = ['query: What is shear stress',
               'query: Why strength of matrtial important',]
# Each input text should start with "query: " or "passage: ".
# For tasks other than retrieval, you can simply use the "query: " prefix.
input_texts = []
tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-small-v2')
model = AutoModel.from_pretrained('intfloat/e5-small-v2').cuda()

# Tokenize the input texts
batch_dict = tokenizer(question, max_length=512, padding=True, truncation=True, return_tensors='pt')
batch_dict = batch_dict.to('cuda')
outputs = model(**batch_dict)
embeddings_q = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])


batch_dict = Tensor(contents).to('cuda')
outputs = model(batch_dict)
embeddings_k = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings_q @ embeddings_k.T) * 5
ranked_indices = np.argsort(scores)[::-1]
K = 5
top_K_indices = ranked_indices[:K]
for i in top_K_indices:
    print(f'{i+1}: {contents[i]}/n')





