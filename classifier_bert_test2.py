import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import pandas as pd
import eyrc18track1test1 as eyrc18
import numpy as np

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

data = pd.read_csv("F:/World of Smita/PiazzaVirtualAssistant-master/PiazzaVirtualAssistant-master/mydata.csv", encoding="latin1")

#print(data)

data = data.fillna(method="ffill")

wd = data["MAIN_CONTENT"].values.tolist()
tg = data["TAGS"].values.tolist()

x1 = wd #np.load("Data/text.npy")
y = tg[1:] #np.load("Data/labels.npy")
x = x1[1:]
#print(x)

tokenized_text = ""

for i in enumerate(x):
    # Tokenized input
    text = x#"Who was Jim Henson ? Jim Henson was a puppeteer"
    tokenized_text = tokenizer.tokenize(str(i))


    masked_index = 15
    tokenized_text[masked_index] = '[MASK]'
    break


print(len(tokenized_text))
print(tokenized_text)
assert tokenized_text[15] == '[MASK]'

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)

segments_ids = []
for i in range(0,55):
    segments_ids.append(0)
for j in range(55,110):
    segments_ids.append(1)
    
print(segments_ids)
print(len(segments_ids))

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])


print(text[0])
#print(tokens_tensor[0])

model = BertModel.from_pretrained('bert-base-uncased')
model.cuda()

# Predict hidden states features for each layer
encoded_layers, _ = model(tokens_tensor, segments_tensors)
# We have a hidden states for each of the 12 layers in model bert-base-uncased
assert len(encoded_layers) == 110


model = BertForMaskedLM.from_pretrained('bert-base-uncased')
print(model)
model.cuda()


# Predict all tokens
predictions = model(tokens_tensor, segments_tensors)

# confirm we were able to predict 'henson'
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
assert predicted_token == 'has'
