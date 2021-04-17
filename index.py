import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import clean_sentence, read_data
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import svm
data_train = read_data('train_pairs.csv')
data_test = read_data('test_pairs.csv')
# a[start:stop:step]
corpus = data_train[:, 0:2:1].flatten()
corpus = np.append(corpus, data_test[:, 0:2].flatten())
clean_corpus = [clean_sentence(doc) for doc in corpus]
vectorizer = TfidfVectorizer()
vectorizer.fit(clean_corpus)

def test_step(vectorTrain):
    print(vectorTrain)
    origins_vec = vectorizer.transform(data_test[:, 0])
    suspects_vec = vectorizer.transform(data_test[:, 1])
    labels = data_test[:, 2]
    clf = svm.SVC()
    data = []
    label = []
    for x in vectorTrain:
        data.append(x[0][0])
        label.append(x[1])
    print(data)
    clf.fit(data, label)
    score = 0
    for origin_vec, suspect_vec, label in zip(origins_vec, suspects_vec, labels):
        sim = cosine_similarity(origin_vec, suspect_vec)
        result = clf.predict(sim)[0]
        if result == label:
            score += 1

    accurate = score/len(data_test)
    print(accurate)


def train_step():
    origins_vec = vectorizer.transform(data_train[:, 0])
    suspects_vec = vectorizer.transform(data_train[:, 1])
    labels = data_train[:, 2]
    vector = []
    for origin_vec, suspect_vec, label in zip(origins_vec, suspects_vec, labels):
        sim = cosine_similarity(origin_vec, suspect_vec)
        x = [sim, label]
        vector.append(x)
    return vector

vectorTrain = train_step()
test_step(vectorTrain)
