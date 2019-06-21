import numpy as np
from tqdm import tqdm, trange
import torch
from sklearn.metrics import f1_score
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForSequenceClassification, BertAdam


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


file = open("result.txt", "w")

x = np.load("Data/text.npy")
y = np.load("Data/labels.npy")

MAX_LEN = 100
bs = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

tokenized_texts = [tokenizer.tokenize(text) for text in x]

input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

attention_masks = [[float(i>0) for i in ii] for ii in input_ids]

encoder = LabelEncoder()
labels = encoder.fit_transform(y)
n_classes = len(set(labels))

tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, labels,
                                                            random_state=2018, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)

tr_inputs = torch.LongTensor(tr_inputs)
val_inputs = torch.LongTensor(val_inputs)
tr_tags = torch.LongTensor(tr_tags)
val_tags = torch.LongTensor(val_tags)
tr_masks = torch.LongTensor(tr_masks)
val_masks = torch.LongTensor(val_masks)

train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)


model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=n_classes)

model.cuda()

FULL_FINETUNING = True
if FULL_FINETUNING:
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
else:
    param_optimizer = list(model.classifier.named_parameters())
    optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

epochs = 50
max_grad_norm = 1.0
e = 0

for _ in trange(epochs, desc="Epoch"):
    # TRAIN loop
    model.train()
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    predictions, true_labels = [], []
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # forward pass
        loss = model(b_input_ids, token_type_ids=None,
                     attention_mask=b_input_mask, labels=b_labels)
        # backward pass
        loss.backward()
        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        model.zero_grad()

        with torch.no_grad():
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        [predictions.append(p) for p in list(np.argmax(logits, axis=1))]
        [true_labels.append(l) for l in list(label_ids)]

        tmp_tr_accuracy = flat_accuracy(logits, label_ids)

        tr_accuracy += tmp_tr_accuracy

    # print train loss per epoch
    e += 1
    file.write("-------------------------------------epoch : {}-------------------------------------\n".format(e))
    print("Train loss: {}".format(tr_loss / nb_tr_steps))
    file.write("Train loss: {}\n".format(tr_loss / nb_tr_steps))
    print("Training Accuracy: {}".format(tr_accuracy / nb_tr_steps))
    file.write("Training Accuracy: {}\n".format(tr_accuracy / nb_tr_steps))
    # print(predictions)
    # print(true_labels)
    print("Training F1: {}".format(f1_score(true_labels, predictions, average="micro")))
    file.write("Training F1: {}\n".format(f1_score(true_labels, predictions, average="micro")))
    # VALIDATION on validation set
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions, true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # print(label_ids)
        # print(np.argmax(logits, axis=1))
        # input()
        [predictions.append(p) for p in list(np.argmax(logits, axis=1))]
        [true_labels.append(l) for l in list(label_ids)]

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    print("Validation loss: {}".format(eval_loss))
    file.write("Validation loss: {}\n".format(eval_loss))
    print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
    file.write("Validation Accuracy: {}\n".format(eval_accuracy / nb_eval_steps))
    print("Validation F1: {}".format(f1_score(true_labels, predictions, average="micro")))
    file.write("Validation F1: {}\n".format(f1_score(true_labels, predictions, average="micro")))
    # pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
    # valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
    # print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))

file.close()