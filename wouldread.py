import gzip
from collections import defaultdict
import random as rand

import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score

import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

def readGz(path):
  for l in gzip.open(path, 'rt'):
    yield eval(l)

def readCSV(path):
  f = gzip.open(path, 'rt')
  f.readline()
  for l in f:
    yield l.strip().split(',')

# Would-read baseline: just rank which books are popular and which are not, and return '1' if a book is among the top-ranked

bookCount = defaultdict(int)
totalRead = 0

userRead = defaultdict(list)
bookAudience = defaultdict(list)
bookSet = set()
X = []

for user,book,_ in readCSV("train_Interactions.csv.gz"):
  bookCount[book] += 1
  totalRead += 1

  # we are given that there are 200000 examples
  if totalRead <= 190000:
    userRead[user].append(book)
    bookAudience[book].append(user)
  bookSet.add(book)
  X.append([user, book, True])

N = len(X)
X_val = X

X_val_neg = []

for ex in X_val:
  user = ex[0]
  new_book = rand.sample(bookSet, 1)[0]
  while new_book in userRead[user]:
    new_book = rand.sample(bookSet, 1)[0]
  X_val_neg.append([user, new_book, False])

X_val = X_val + X_val_neg

def val_pipe(threshold):
  return1 = set()
  count = 0
  # for ic, i in mostPopular:
  #   count += ic
  #   return1.add(i)
  #   if count > threshold*totalRead: break

  val_pred = [ex[1] in return1 for ex in X_val]
  val_gt = [ex[2] for ex in X_val]

  correct = [val_gt[i] == val_pred[i] for i in range(len(val_gt))]
  accuracy = sum(correct) / len(correct)

  print('accuracy: '+str(accuracy))

# for n in range(0,20):
#   val_pipe(0.05*n)

# Jaccard-based pipeline
def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom

def jaccard_pipe(threshold):
  
  val_pred = []

  for ex in X_val:
    max_jac = 0

    bookList = userRead[ex[0]]
    b_base = ex[1]
    s_base = set(bookAudience[b_base])

    for b_prime in bookList:
      s_prime = set(bookAudience[b_prime])
      jacc = Jaccard(s_base, s_prime)
      if jacc > max_jac:
        max_jac = jacc
    
    print(max_jac)
    val_pred.append(max_jac > threshold)

  
  val_gt = [ex[2] for ex in X_val]

  correct = [val_gt[i] == val_pred[i] for i in range(len(val_gt))]
  accuracy = sum(correct) / len(correct)
  print('accuracy at threshold '+str(threshold)+': '+str(accuracy))

# for n in range(0, 20):
#   jaccard_pipe(0.0025*n)

def maxJaccard(user, book):
  max_jac = 0
  bookList = userRead[user]
  s_base = set(bookAudience[book])
  for other_book in bookList:
    s_prime = set(bookAudience[other_book])
    jacc = Jaccard(s_base, s_prime)
    if jacc > max_jac:
      max_jac = jacc
  return max_jac

# popularity-based pipeline
mostPopular = [(bookCount[x], x) for x in bookCount]
mostPopular.sort()
mostPopular.reverse()

bookPop = defaultdict(float)
count = 0
for ic, i in mostPopular:
  count += ic
  bookPop[i] = 1 - (count/totalRead) # append popularity measure to this

print("popular dict filling done")

# training the classifier
constant = np.ones(len(X_val))
popularity = []
jacc_feat = []
for ex in X_val:
  user = ex[0]
  book = ex[1]
  popularity.append(bookPop[book])
  jacc_feat.append(maxJaccard(user, book))

print("feature making done")

popularity = np.array(popularity)
jacc_feat = np.array(jacc_feat)
y = np.array([ex[2] for ex in X_val])

# print(constant.shape)
# print(popularity.shape)
# print(jacc_feat.shape)
# print(y.shape)

Zy = np.stack([constant, popularity, jacc_feat, y]).T
rand.shuffle(Zy)

Z = [d[:-1] for d in Zy]
y = [d[-1] for d in Zy]

M = len(Z)
Z_train = Z[:3*M//4]
Z_val = Z[3*M//4:]
y_train = y[:3*M//4]
y_val = y[3*M//4:]

print("commencing training...")

def metatrain(c):
  model = linear_model.LogisticRegression(C=c, class_weight='balanced',
          solver='lbfgs')
  model.fit(Z_train, y_train)
  
  pred = model.predict(Z_val)
  correct = pred == y_val
  acc = sum(correct) / len(correct)

  print('Accuracy using C value '+str(c)+': '+str(acc))

# for i in [10, 30, 50, 100, 300, 500, 1000, 2000, 3000, 4000, 5000]:
#   metatrain(i)

model = linear_model.LogisticRegression(C=3000, class_weight='balanced')
model.fit(Z_train, y_train)

predictions = open("predictions_Read.txt", 'w')
for l in open("pairs_Read.txt"):
  if l.startswith("userID"):
    # header
    predictions.write(l)
    continue
  u,b = l.strip().split('-')

  sampleList = np.array([1, bookPop[b], maxJaccard(u,b)])
  sampleList = sampleList.reshape(1, -1)
  
  if model.predict(sampleList)[0]:
    predictions.write(u + '-' + b + ",1\n")
  else:
    predictions.write(u + '-' + b + ",0\n")

print('predictions written in predictions_Read.txt!')
predictions.close()