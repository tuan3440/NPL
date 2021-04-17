from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import regex as re
import pandas as pd
import numpy as np

stop_words = set(stopwords.words('english'))

def read_data(file_name):
    df_data = pd.read_csv(file_name)

    origin = df_data['Origin'].to_numpy()
    suspect = df_data['Suspect'].to_numpy()
    label = df_data['Label'].to_numpy().astype('int')
    print(type(label[0]))
    data = []
    for s1, s2, label in zip(origin, suspect, label):
        data.append([s1, s2, label])

    return np.array(data)

def tokenizer(sentence):
    return word_tokenize(sentence)


def clean_sentence(sentence):
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub('', sentence).lower()
    return sentence


def remove_stopwords(token_sentence):
    result = []
    for word in token_sentence:
        if word not in stop_words:
            result.append(word)
    return result

def clean_and_tokenizer_text(sentence):
    sentence = clean_sentence(sentence)
    sentence = tokenizer(sentence)
    return sentence


# from nltk.metrics import edit_distance
# from nltk import pos_tag
#
# sentence = 'million is billion'
# print(pos_tag(tokenizer(sentence)))