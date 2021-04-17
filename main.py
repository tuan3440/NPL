import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import clean_sentence, read_data
from sklearn.metrics.pairwise import cosine_similarity
data_train = read_data('train_pairs.csv')
data_test = read_data('test_pairs.csv')
# a[start:stop:step]
corpus = data_train[:, 0:2:1].flatten()
corpus = np.append(corpus, data_test[:, 0:2].flatten())
clean_corpus = [clean_sentence(doc) for doc in corpus]
vectorizer = TfidfVectorizer()
vectorizer.fit(clean_corpus)

def test_step(threshold):
    origins_vec = vectorizer.transform(data_test[:, 0])
    suspects_vec = vectorizer.transform(data_test[:, 1])
    labels = data_test[:, 2]

    score = 0
    accuracy = 0
    for origin_vec, suspect_vec, label in zip(origins_vec, suspects_vec, labels):
        sim = cosine_similarity(origin_vec, suspect_vec)
        if sim > threshold:
            if float(label) == 1:
                score += 1
    accuracy = score / len(data_test)

    print('Accuracy test:', accuracy)


def train_step():
    origins_vec = vectorizer.transform(data_train[:, 0])
    suspects_vec = vectorizer.transform(data_train[:, 1])
    labels = data_train[:, 2]
    array_t = [0.5, 0.6, 0.7, 0.8, 0.9]

    best_accuracy = 0
    best_t = 0
    for t in array_t:
        accuracy = 0
        score = 0
        for origin_vec, suspect_vec, label in zip(origins_vec, suspects_vec, labels):
            sim = cosine_similarity(origin_vec, suspect_vec)
            if sim > t:
                if float(label) == 1:
                    score += 1

        accuracy = score / len(data_train)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_t = t

    print('Best t:', best_t)
    return best_t

threshold = train_step()
test_step(threshold)
