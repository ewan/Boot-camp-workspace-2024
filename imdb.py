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

reviews_glove_pooled_l = []
for review in reviews_glove:
    reviews_glove_pooled_l.append(np.average(review, axis=0))
reviews_glove_pooled = np.concatenate(reviews_glove_pooled_l).reshape((-1,50))

dtype = torch.float
torch.set_default_device("cpu")
X = torch.tensor(reviews_glove_pooled)
y = torch.tensor([{"positive": 1, "negative": 0}[x] for x in imdb['sentiment']], dtype=dtype)

lin1 = torch.nn.Linear(50, 25, dtype=dtype)
lin2 = torch.nn.Linear(25, 1, dtype=dtype)

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(list(lin1.parameters()) + list(lin2.parameters()), lr=0.001)

print("Optimizing...")
for e in range(2000):
    optimizer.zero_grad()
    h1 = torch.nn.functional.sigmoid(lin1(X))
    z = lin2(h1).squeeze()
    loss = loss_fn(z, y)
    if (e % 25 == 1):
        print(e, loss.item(), accuracy(z, y))
    loss.backward()
    optimizer.step()

