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
        vector = np.array(line.split()[1:], dtype="float16")
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
torch.set_default_device("cpu")
#X = torch.tensor(reviews_glove_pooled, dtype=dtype)
y = torch.tensor([{"positive": 1, "negative": 0}[x] for x in imdb['sentiment']], dtype=dtype)

hid_size = 128

rnn = torch.nn.RNN(50, hid_size, dtype=dtype)
M = torch.nn.Linear(hid_size, 1, dtype=dtype)

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(list(rnn.parameters()) + list(M.parameters()), lr=0.001)

def do_rnn(X):
    z = rnn(X).squeeze()
    return z

print("Optimizing...")
for e in range(2000):
    indices = list(range(len(reviews_glove)))
    np.random.shuffle(indices)
    sum_of_loss_for_printing = 0
    for i in indices:
        optimizer.zero_grad()
        X_i = torch.tensor(reviews_glove[i], dtype=dtype)
        z = do_rnn(X_i)
        loss = loss_fn(z, y[i].squeeze())
        sum_of_loss_for_printing += loss.item()
        loss.backward()
        optimizer.step()
    print(e, sum_of_loss_for_printing/len(indices))

