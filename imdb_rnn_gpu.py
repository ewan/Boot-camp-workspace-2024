import pandas as pd
import re
import numpy as np
import torch

print("Starting...")

def accuracy(pred_z, true):
    with torch.no_grad():
        return (pred_z*(true - 0.5) >= 0).float().mean()

def process(x):
    x = re.sub('[,\.!?:()"]', '', x)
    x = re.sub('<.*?>', ' ', x)
    x = re.sub('http\S+', ' ', x)
    x = re.sub('[^a-zA-Z0-9]', ' ', x)
    x = re.sub('\s+', ' ', x)
    return x.lower().strip()

imdb = pd.read_csv("IMDB.csv")
reviews = [process(x) for x in imdb['review'].tolist()]

glove = {}
with open("glove_imdb.6B.50d.txt") as hf:
    for line in hf:
        word = line.split()[0]
        vector = np.array(line.split()[1:], dtype="float")
        glove[word] = vector

print("Converting reviews to glove...")
reviews_glove = []
for review in reviews:
    review_words = review.split()
    review_glove_words = []
    for word in review_words:
        try:
            review_glove_words.append(glove[word])
        except KeyError:
            continue
    review_glove_np = np.concatenate(review_glove_words).reshape((-1,50))
    reviews_glove.append(review_glove_np)

dtype = torch.float
device = torch.device("cuda:0")
#device = torch.device("cpu" )
#X = torch.tensor(reviews_glove_pooled, dtype=dtype)
y_train = torch.tensor([{"positive": 1, "negative": 0}[x] for x in imdb['sentiment']], dtype=dtype)

hid_size = 256

rnn = torch.nn.RNN(50, hid_size, dtype=dtype, device=device)
M = torch.nn.Linear(hid_size, 1, dtype=dtype, device=device)

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(list(rnn.parameters()) + list(M.parameters()), lr=0.01)

def do_rnn(X):
    z = M(rnn(X)[1]).squeeze()
    return z

class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, review_list, label_tensor):
        self.review_list = review_list
        self.label_tensor = label_tensor
    
    def __len__(self):
        return len(self.review_list)
    
    def __getitem__(self, idx):
        X = self.review_list[idx]
        y = self.label_tensor[idx]
        return X.to(device), y.to(device)

def collate_fn(data):
    X, y = zip(*data)
    lengths = [len(x) for x in X]
    xy_sorted = [(x, lab) for _, x, lab in sorted(zip(lengths, X, y),
                                      key=lambda x: x[0],
                                      reverse=True)]
    X_sorted, y_sorted = zip(*xy_sorted)
    X_ps = torch.nn.utils.rnn.pack_sequence(X_sorted)
    y_t = torch.stack(y_sorted).squeeze()
    return X_ps, y_t

reviews_glove_t = [torch.tensor(r, dtype=dtype) for r in reviews_glove]
batch_size = 256
loader = torch.utils.data.DataLoader(
    ReviewDataset(reviews_glove_t, y_train),
    batch_size=batch_size,
    collate_fn=collate_fn,
    shuffle=True,
    drop_last=True
)


print_time = 50
print("Optimizing...")
for e in range(10):
    sum_of_loss_for_printing = 0
    for t, (X,y) in enumerate(loader):
        optimizer.zero_grad()
        z = do_rnn(X)
        loss = loss_fn(z, y)
        sum_of_loss_for_printing += loss.item()
        if ((t+1) % print_time == 0):
            print(e, t*batch_size, sum_of_loss_for_printing/print_time)
            sum_of_loss_for_printing = 0
        loss.backward()
        optimizer.step()

with torch.no_grad():
    n_correct = 0
    for t, (X, y) in enumerate(loader):
        z = do_rnn(X)
        pred = torch.squeeze(z).cpu().detach().numpy() > 0
        y_np = torch.squeeze(y).cpu().detach().numpy()
        n_correct += (pred == y_np).sum()
print("Train accuracy: ", n_correct/len(reviews_glove))

