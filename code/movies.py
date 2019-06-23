import pandas as pd
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import re
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import nltk
from sklearn.metrics import accuracy_score, f1_score
file = pd.read_csv('tagged_plots_movielens.csv')
file = file.dropna()
file['plot'].apply(lambda x: len(x.split(' '))).sum()

my_tags = file.tag.unique()
file.tag.value_counts().plot(kind="bar", rot=0)
trainedData, testedData = train_test_split(file, test_size=0.1, random_state=42)
testedData.tag.value_counts().plot(kind="bar", rot=0)
model = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0)


def movie_genre(index):
    example = file[file.index == index][['plot', 'tag']].values[0]
    if len(example) > 0:
        print(example[0])
        print('Genre:', example[1])


def cleanText(text):
    text = text.lower()
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text


def tokenize_text(text):
    list_ = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            list_.append(word.lower())
    return list_


file['plot'] = file['plot'].apply(cleanText)
trainedTag = trainedData.apply(lambda r: TaggedDocument(words=tokenize_text(r['plot']), tags=[r.tag]), axis=1)
testTag = testedData.apply(lambda r: TaggedDocument(words=tokenize_text(r['plot']), tags=[r.tag]), axis=1)
model.build_vocab([x for x in tqdm(trainedTag.values)])
for i in range(30):
    model.train(utils.shuffle([x for x in tqdm(trainedTag.values)]), total_examples=len(trainedTag.values), epochs=1)
    model.alpha -= 0.002
    model.min_alpha = model.alpha


def vector(model, documentTag):
    sents = documentTag.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors


model_train, tag_train = vector(model, trainedTag)
model_test, tag_test = vector(model, testTag)

lr = LogisticRegression(n_jobs=1, C=1e5)
lr.fit(tag_train, model_train)
model_prediction = lr.predict(tag_test)


movie_genre(5)
print('accuracy %s' % accuracy_score(model_test, model_prediction))