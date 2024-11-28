import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace
import pandas as pd
import pickle
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score

from tokenizer import BertTokenizer
from bert import BertModel
from extras_utils import get_author_embedding, get_authors_embedding
# from optimizer import AdamW
from torch.optim import AdamW
from tqdm import tqdm
import pandas as pd

import pandas as pd


TQDM_DISABLE=False
# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class BertSentClassifier(torch.nn.Module):
    def __init__(self, config):
        super(BertSentClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained('google/bert_uncased_L-4_H-256_A-4', config.pretrained_bert_file, use_checkpoint=config.use_checkpoint)

        # pretrain mode does not require updating bert paramters.
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True

        self.use_author = config.use_author

        if config.use_author:
            extra_dim = config.author_size
        else:   
            extra_dim = 0
        
        # todo
        # raise NotImplementedError
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size + extra_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, config.num_labels)
        )
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, attention_mask, author_embedding):
        output = self.bert(input_ids, attention_mask)
        pooled = output['pooler_output']

        if self.use_author:
            pooled = torch.cat((pooled, author_embedding), 1)
            
        # return self.softmax(self.project(self.dropout(pooled)))
        return self.softmax(self.mlp(self.dropout(pooled)))

# create a custom Dataset Class to be used for the dataloader
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
        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        author_embedding = torch.stack([x[3] for x in data], dim=0)
        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True, max_length=512)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        token_type_ids = torch.LongTensor(encoding['token_type_ids'])
        labels = torch.LongTensor(labels)
        

        return token_ids, token_type_ids, attention_mask, labels, sents, author_embedding

    def collate_fn(self, all_data):
        all_data.sort(key=lambda x: -len(x[2]))  # sort by number of tokens

        batches = []
        num_batches = int(np.ceil(len(all_data) / self.p.batch_size))

        for i in range(num_batches):
            start_idx = i * self.p.batch_size
            data = all_data[start_idx: start_idx + self.p.batch_size]

            token_ids, token_type_ids, attention_mask, labels, sents, author_embedding = self.pad_data(data)
            batches.append({
                'token_ids': token_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'sents': sents,
                'author_embedding': author_embedding
            })

        return batches

def create_data(filename, author2embedding_filename='data/author2embedding.pickle', flag='train'):

    # specify the tokenizer
    tokenizer = BertTokenizer.from_pretrained('google/bert_uncased_L-4_H-256_A-4')
    num_labels = {}
    data = []

    with open(author2embedding_filename, 'rb') as f:
        author2embedding = pickle.load(f)

    label_map = {
        "Children's literature" : 0,
        "Crime Fiction" : 1,
        "Fantasy" : 2,
        "Mystery" : 3,
        "Non-fiction" : 4,
        "Science Fiction" : 5,
        "Suspense" : 6,
        "Young adult literature" : 7
    }

    df = pd.read_csv(filename)
    for _, row in df.iterrows():
        label = label_map[row['Genres']]
        if label not in num_labels:
            num_labels[label] = len(num_labels)
        sent = row['Summary'].lower().strip()
        tokens = tokenizer.tokenize("[CLS] " + sent + " [SEP]")
        book_authors = row['Book Author']
        author_embedding = get_authors_embedding(author2embedding, book_authors)
        data.append((sent, label, tokens, author_embedding))
    print(f"load {len(data)} data from {filename}")
    if flag == 'train':
        return data, len(num_labels)
    else:
        return data

def model_eval(dataloader, model, device):
    model.eval() # switch to eval model, will turn off randomness like dropout
    y_true = []
    y_pred = []
    sents = []
    total_loss = 0.0
    num_batches = 0
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids, b_type_ids, b_mask, b_labels, b_sents, b_author_embedding = batch[0]['token_ids'], batch[0]['token_type_ids'], \
                                                       batch[0]['attention_mask'], batch[0]['labels'], batch[0]['sents'], batch[0]['author_embedding']  

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)
        b_author_embedding = b_author_embedding.to(device)
        b_labels = b_labels.to(device)

        logits = model(b_ids, b_mask, b_author_embedding)

        with torch.no_grad():
            loss = F.nll_loss(logits, b_labels.view(-1), reduction='sum') / args.batch_size

        total_loss += loss.item()
        num_batches += 1

        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        b_labels = b_labels.cpu().numpy().flatten()
        y_true.extend(b_labels)
        y_pred.extend(preds)
        sents.extend(b_sents)

    total_loss = total_loss / num_batches
    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)

    return acc, f1, total_loss, y_pred, y_true, sents

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
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    #### Load data
    # create the data and its corresponding datasets and dataloader
    train_data, num_labels = create_data(args.train,args.author2embedding_filename,flag='train')
    dev_data = create_data(args.dev,args.author2embedding_filename, flag='valid')

    train_dataset = BertDataset(train_data, args)
    dev_dataset = BertDataset(dev_data, args)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 256,
              'data_dir': '.',
              'author_size': 200,
              'use_author': args.use_author,
              'option': args.option,
              'author2embedding_filename': args.author2embedding_filename,
              'pretrained_bert_file': args.pretrained_bert_file,
              'use_checkpoint': args.use_checkpoint,
              'filepath': args.filepath,
              }

    config = SimpleNamespace(**config)

    model = BertSentClassifier(config)
    if args.pretrained_model is not None:
        save = torch.load(args.pretrained_model)
        print('Loading model from: ', args.pretrained_model)
        model.load_state_dict(save['model'])
        print('Model loaded from: ', args.pretrained_model)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    run_filepath = f'{args.output_dir}/run.csv'

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)):
            b_ids, b_type_ids, b_mask, b_labels, b_sents, b_author_embedding = batch[0]['token_ids'], batch[0]['token_type_ids'], batch[0][
                'attention_mask'], batch[0]['labels'], batch[0]['sents'], batch[0]['author_embedding']

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_author_embedding = b_author_embedding.to(device)
            b_labels = b_labels.to(device)

            optimizer.zero_grad()
            logits = model(b_ids, b_mask, b_author_embedding)
            loss = F.nll_loss(logits, b_labels.view(-1), reduction='sum') / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_acc, train_f1, eval_train_loss, *_ = model_eval(train_dataloader, model, device)
        dev_acc, dev_f1, dev_loss, *_ = model_eval(dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.save_model_path)

        print(f"epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev loss :: {dev_loss :.3f}, dev acc :: {dev_acc :.3f}")
        
        # Check if the file already exists to determine whether to write the header
        file_exists = os.path.isfile(run_filepath)
        with open(run_filepath, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            if not file_exists:
                writer.writerow(['epoch', 'train_loss', 'train_acc', 'dev_loss', 'dev_acc'])
            writer.writerow([epoch, train_loss, train_acc, dev_loss, dev_acc])


def test(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.save_model_path)
        config = saved['model_config']
        model = BertSentClassifier(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"load model from {args.save_model_path}")
        dev_data = create_data(args.dev, flag='valid')
        dev_dataset = BertDataset(dev_data, args)
        dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

        test_data = create_data(args.test, flag='test')
        test_dataset = BertDataset(test_data, args)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)

        dev_acc, dev_f1, dev_loss, dev_pred, dev_true, dev_sents = model_eval(dev_dataloader, model, device)
        test_acc, test_f1, test_loss, test_pred, test_true, test_sents = model_eval(test_dataloader, model, device)

        dev_out = f'{args.output_dir}/dev_result.txt'
        test_out = f'{args.output_dir}/test_result.txt'

        with open(dev_out, "w+", encoding="utf-8") as f:
            print(f"dev acc :: {dev_acc :.3f}")
            for s, t, p in zip(dev_sents, dev_true, dev_pred):
                f.write(f"{s} ||| {t} ||| {p}\n")

        with open(test_out, "w+", encoding="utf-8") as f:
            print(f"test acc :: {test_acc :.3f}")
            for s, t, p in zip(test_sents, test_true, test_pred):
                f.write(f"{s} ||| {t} ||| {p}\n")

        # Define label map
        label_map = {
            0: "Children's literature",
            1: "Crime Fiction",
            2: "Fantasy",
            3: "Mystery",
            4: "Non-fiction",
            5: "Science Fiction",
            6: "Suspense",
            7: "Young adult literature"
        }
        
        # Generate classification report
        dev_report = classification_report(dev_true, dev_pred, target_names=label_map.values(), labels=list(label_map.keys()), digits=3, zero_division=0)
        test_report = classification_report(test_true, test_pred, target_names=label_map.values(), labels=list(label_map.keys()), digits=3, zero_division=0)

        # Save classification report for dev
        dev_report_path = f'{args.output_dir}/dev_report.txt'
        with open(dev_report_path, "w+", encoding="utf-8") as f:
            f.write("Classification Report for Dev Data\n")
            f.write(dev_report)

        # Save classification report for test
        # Save classification report for test
        test_report_path = f'{args.output_dir}/test_report.txt'
        with open(test_report_path, "w+", encoding="utf-8") as f:
            f.write("Classification Report for Test Data\n")
            f.write(test_report)
        print(f"Test report saved to {test_report_path}")

        # Print classification reports for reference
        print("Classification Report for Dev Data:")
        print(dev_report)
        
        print("Classification Report for Test Data:")
        print(test_report)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="data/cfimdb-train.txt")
    parser.add_argument("--dev", type=str, default="data/cfimdb-dev.txt")
    parser.add_argument("--test", type=str, default="data/cfimdb-test.txt")
    parser.add_argument("--author2embedding_filename", type=str, default="data/author2embedding.pickle")
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--pretrained_model", type=str, default=None)
    parser.add_argument("--save_model_path", type=str, default=None)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--use_author", action='store_true')
    parser.add_argument("--output_dir", type=str, default="output")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)

    parser.add_argument("--filepath", type=str, default="kaggle/working")
    parser.add_argument("--pretrained_bert_file", type=str, default="google/bert_uncased_L-4_H-256_A-4")
    parser.add_argument("--use_checkpoint", action='store_true')
    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args

if __name__ == "__main__":
    args = get_args()

    # Ensure the directory exists
    output_folder = f'{args.output_dir}/'
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    if args.save_model_path is None:
        args.save_model_path = f'{args.output_dir}/{args.option}-{args.epochs}-{args.lr}.pt' # save path
    else:
        args.save_model_path = f'{args.output_dir}/{args.save_model_path}' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train(args)
    test(args)
