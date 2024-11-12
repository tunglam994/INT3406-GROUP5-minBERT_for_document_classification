import time, random, numpy as np, argparse, sys, re, os
import pandas as pd
from types import SimpleNamespace
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score

from tokenizer import BertTokenizer
from bert import BertModel
# from optimizer import AdamW
from torch.optim import AdamW
from tqdm import tqdm
import torch.nn as nn

TQDM_DISABLE=False
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class Loss():

    def __init__(self, args):
        self.args = args
        self.cos = nn.CosineSimilarity(dim=-1)

    def train_loss_fct(self, criterion, a, p, n, neg_weight=0):
        device = torch.device('cuda') if self.args.use_gpu else torch.device('cpu')
        positive_similarity = self.cos(a.unsqueeze(1), p.unsqueeze(0)) / self.args.temperature
        negative_similarity = self.cos(a.unsqueeze(1), n.unsqueeze(0)) / self.args.temperature
        
        cosine_similarity = torch.cat([positive_similarity, negative_similarity], dim=1).to(device)

        labels = torch.arange(cosine_similarity.size(0)).long().to(device)

        weights = torch.tensor(
            [[0.0] * (cosine_similarity.size(-1) - negative_similarity.size(-1)) + [0.0] * i + [neg_weight] + [0.0] * (negative_similarity.size(-1) - i - 1) for i in range(negative_similarity.size(-1))]
        ).to(device)

        cosine_similarity = cosine_similarity + weights
        loss = criterion(cosine_similarity, labels)

        return loss
    

class BertBooksClassifier(torch.nn.Module):
    def __init__(self, config):
        super(BertBooksClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('google/bert_uncased_L-4_H-256_A-4')

        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask)
        pooled = output['pooler_output']
        return self.dropout(pooled)

class BertDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-4_H-256_A-4')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ele = self.dataset[idx]
        return ele

    def pad_data(self, data):
        anchor = [x[0] for x in data]
        pos = [x[2] for x in data]
        neg = [x[4] for x in data]

        encoding_anchor = self.tokenizer(anchor, return_tensors='pt', padding=True, truncation=True)
        token_ids_anchor = torch.LongTensor(encoding_anchor['input_ids'])
        attention_mask_anchor = torch.LongTensor(encoding_anchor['attention_mask'])

        encoding_pos = self.tokenizer(pos, return_tensors='pt', padding=True, truncation=True)
        token_ids_pos = torch.LongTensor(encoding_pos['input_ids'])
        attention_mask_pos = torch.LongTensor(encoding_pos['attention_mask'])

        encoding_neg = self.tokenizer(neg, return_tensors='pt', padding=True, truncation=True)
        token_ids_neg = torch.LongTensor(encoding_neg['input_ids'])
        attention_mask_neg = torch.LongTensor(encoding_neg['attention_mask'])

        return token_ids_anchor, attention_mask_anchor, \
                token_ids_pos, attention_mask_pos, \
                token_ids_neg, attention_mask_neg

    def collate_fn(self, all_data):
        all_data.sort(key=lambda x: -len(x[1]))  # sort by number of tokens

        batches = []
        num_batches = int(np.ceil(len(all_data) / self.p.batch_size))

        for i in range(num_batches):
            start_idx = i * self.p.batch_size
            data = all_data[start_idx: start_idx + self.p.batch_size]

            token_ids_anchor, attention_mask_anchor, \
                token_ids_pos, attention_mask_pos, \
                token_ids_neg, attention_mask_neg = self.pad_data(data)
            batches.append({
                'token_ids_anchor': token_ids_anchor,
                'attention_mask_anchor': attention_mask_anchor,
                'token_ids_pos': token_ids_pos,
                'attention_mask_pos': attention_mask_pos,
                'token_ids_neg': token_ids_neg,
                'attention_mask_neg': attention_mask_neg,
            })

        return batches

def create_data(filename, flag='train'):
    tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-4_H-256_A-4')
    num_labels = {}
    data = []

    df = pd.read_csv(filename)
    for _, row in df.iterrows():
        # id = row['id']
        anchor = row['sent0'].lower()
        pos = row['sent1'].lower()
        neg = row['hard_neg'].lower()

        tokens_anchor = tokenizer.tokenize("[CLS] " + anchor + " [SEP]")
        tokens_pos = tokenizer.tokenize("[CLS] " + pos + " [SEP]")
        tokens_neg = tokenizer.tokenize("[CLS] " + neg + " [SEP]")
        data.append((anchor, tokens_anchor, pos, tokens_pos, neg, tokens_neg))
    print(f"load {len(data)} data from {filename}")
    return data

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def train(args):
    loss = Loss(args)
    criterion = nn.CrossEntropyLoss()

    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    train_data = create_data(args.train, 'train')
    dev_data = create_data(args.dev, 'valid')

    train_dataset = BertDataset(train_data, args)
    dev_dataset = BertDataset(dev_data, args)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    #### Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
            #   'num_labels': num_labels,
              'hidden_size': 256,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    # initialize the Senetence Classification Model
    model = BertBooksClassifier(config)
    if args.pretrained_model is not None:
        save = torch.load(args.pretrained_model)
        print('Loading model from: ', args.pretrained_model)
        model.load_state_dict(save['model'])
        print('Model loaded from: ', args.pretrained_model)
    model = model.to(device)

    lr = args.lr
    ## specify the optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
            b_ids_anchor, b_mask_anchor, \
            b_ids_pos, b_mask_pos, \
            b_ids_neg, b_mask_neg  = batch[0]['token_ids_anchor'], batch[0]['attention_mask_anchor'], \
                                    batch[0]['token_ids_pos'], batch[0]['attention_mask_pos'], \
                                    batch[0]['token_ids_neg'], batch[0]['attention_mask_neg']

            b_ids_anchor = b_ids_anchor.to(device)
            b_mask_anchor = b_mask_anchor.to(device)
            b_ids_pos = b_ids_pos.to(device)
            b_mask_pos = b_mask_pos.to(device)
            b_ids_neg = b_ids_neg.to(device)
            b_mask_neg = b_mask_neg.to(device)
            
            optimizer.zero_grad()

            anchor_pooler = model(b_ids_anchor, b_mask_anchor)
            pos_pooler = model(b_ids_pos, b_mask_pos)
            neg_pooler = model(b_ids_neg, b_mask_neg)
            batch_loss = loss.train_loss_fct(criterion, anchor_pooler, pos_pooler, neg_pooler)

            optimizer.zero_grad()
            batch_loss.backward()

            optimizer.step()

            train_loss += batch_loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)
        print(f"epoch {epoch}: train loss :: {train_loss :.3f}")
    save_model(model, optimizer, args, config, args.filepath)


def test(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']
        model = BertBooksClassifier(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"load model from {args.filepath}")
        dev_data = create_data(args.dev, 'valid')
        dev_dataset = BertDataset(dev_data, args)
        dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

        test_data = create_data(args.test, 'test')
        test_dataset = BertDataset(test_data, args)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)

        dev_acc, dev_f1, dev_pred, dev_true, dev_sents = model_eval(dev_dataloader, model, device)
        test_acc, test_f1, test_pred, test_true, test_sents = model_eval(test_dataloader, model, device)

        with open(args.dev_out, "w+") as f:
            print(f"dev acc :: {dev_acc :.3f}")
            for s, t, p in zip(dev_sents, dev_true, dev_pred):
                f.write(f"{s} ||| {t} ||| {p}\n")

        with open(args.test_out, "w+") as f:
            print(f"test acc :: {test_acc :.3f}")
            for s, t, p in zip(test_sents, test_true, test_pred):
                f.write(f"{s} ||| {t} ||| {p}\n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="data/cfimdb-train.txt")
    parser.add_argument("--dev", type=str, default="data/cfimdb-dev.txt")
    parser.add_argument("--test", type=str, default="data/cfimdb-test.txt")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--dev_out", type=str, default="cfimdb-dev-output.txt")
    parser.add_argument("--test_out", type=str, default="cfimdb-test-output.txt")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)
    parser.add_argument("--temperature", type=float, help="temperature for supervised CL", default=0.05)
    parser.add_argument("--filepath", type=str, default="kaggle/working")

    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args

if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)  # fix the seed for reproducibility
    train(args)
    # test(args)