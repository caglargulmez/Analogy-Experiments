from gensim.models import KeyedVectors
import numpy as np
from numpy.linalg import norm
import pandas as pd
import re
import nltk

from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm



words = {}
gomulu = {}
lines = {}

def cleaning_plot(plot):
    # cleaning text in plot with beautifulsoup4
    plot = plot.lower()
    plot = BeautifulSoup(plot, "lxml").text
    plot = re.sub(r'\|\|\|', r' ', plot)
    return plot

def token(plot):

    # make tokenization on plot text
    tokenizes = []

    for sents in nltk.sent_tokenize(plot):
        for element in nltk.word_tokenize(sents):
            if len(element) < 2:
                continue
            tokenizes.append(element.lower())
    return tokenizes

def learning_vectors(model, tagged_docs):
    # returns target vector and regressors
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

def part1():

    # implementation of part 1

    # open test file
    # and read one necessary line
    file = open('word-test.v1.txt', 'r')
    file.readline()
    group = ''              # group name
    for line in file.readlines():

        line = line.replace('\t', '').replace('\n', '').split(' ')

        if line[0] == ':':
            # if line involves group name, create a group in dictionary and set
            words[line[1]] = set()
            group = line[1]
            lines[line[1]] = []
        else:
            # if line involves test data, add it to relevant group
            for i in range(4):
                words[group].add(line[i])
            lines[group].append(line)


    # pre-trained word2vec vectors vocabulary
    vectorss = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    for word, vectors in zip(vectorss.vocab, vectorss.vectors):
        c = np.asarray(vectors, dtype='float32')
        gomulu[word] = c


    # Predicting test data
    for element in words:
        counter1 = 0
        correct = 0
        pred = ''
        for line in lines[element]:
            counter1 = counter1 + 1  # line counter
            max = 0

            # V1 - V0 + V2
            # Vector subtraction before addition
            result = np.add(np.subtract(gomulu[line[1]], gomulu[line[0]]), gomulu[line[2]])

            for unique in words[element]:
                # Cosine similarity to find nearest unique word
                sim = np.dot(result, gomulu[unique]) / (norm(result) * norm(gomulu[unique]))
                if sim > max and unique != line[0] and unique != line[1] and unique != line[2]:
                    max = sim
                    pred = unique

            # if prediction correct
            if pred == line[3]:
                correct = correct + 1


        # displaying accuracy ratios for each group
        print('Group:', element,
              '\nCorrect Prediction:', correct,
              '\nTest Size:', counter1,
              '\nAccuracy Ratio:', correct/counter1,
              '\n##########################')




def part2():

    nltk.download('punkt')

    # read csv file and create dataframe using pandas library
    dataframe = pd.read_csv('tagged_plots_movielens.csv', na_filter=False)
    dataframe = dataframe[['movieId', 'plot', 'tag']]

    # apply cleaning_plot function to dataframe's plot column
    # clean the column from unnecessary details
    dataframe['plot'] = dataframe['plot'].apply(cleaning_plot)

    # splitting train and test data, 2000 train, 448 test
    train_data, test_data = train_test_split(dataframe, test_size=0.1829, shuffle=False)

    # obtaining tagged documents of cleaned train and test datas
    train_tagged = train_data.apply(
        lambda r: TaggedDocument(words=token(r['plot']), tags=[r.tag]), axis=1)
    test_tagged = test_data.apply(
        lambda r: TaggedDocument(words=token(r['plot']), tags=[r.tag]), axis=1)

    # creating doc2vec model
    model_dmm = Doc2Vec(dm=1, dm_mean=1, vector_size=150, window=10, negative=5, min_count=1, workers=5, alpha=0.065,
                        min_alpha=0.0065)

    # initializing LogisticRegression
    regression = LogisticRegression(n_jobs=1, C=1e5)

    # building model vocabulary with tqdm
    model_dmm.build_vocab([x for x in tqdm(train_tagged.values)])

    # training step with 10 epoch, 0.01 learning rate
    for epoch in range(10):
        model_dmm.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values),
                        epochs=1)
        model_dmm.alpha -= 0.01
        model_dmm.min_alpha = model_dmm.alpha


    y_t, X_t = learning_vectors(model_dmm, train_tagged)
    y_te, X_te = learning_vectors(model_dmm, test_tagged)
    regression.fit(X_t, y_t)
    y_pr = regression.predict(X_te)
    print('Accuracy Ratio:', accuracy_score(y_te, y_pr))



part1()

part2()