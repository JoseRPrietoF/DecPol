import os,csv
import numpy as np
from nltk.tokenize import RegexpTokenizer
import nltk, textblob
from textblob.classifiers import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle


pattern = r'\w+'
tokenizerAlpha = RegexpTokenizer(pattern)


"""Devuelve X (muestras) e y(etiquetas)"""
def read_data(path="data/"):
    lst = os.listdir(path)
    lst.sort()
    X = []
    y = []
    for file_name in lst:
        if not file_name.endswith(".csv"):
            continue
        filepath = os.path.join(path, file_name)
        print("Filepath %s " % filepath)
        with open(filepath, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            print(spamreader)
            for row in spamreader:
                if row[1] == '': continue
                X.append(procesar(row[1]))
                y.append(row[2])
    return X,y

"""Devuelve los terminos"""
def procesar(cadena):

    #pasamos to do a minus
    cadena = cadena.lower()
    # Quitamos simbolos no alfanumeicos
    terminos = nltk.regexp_tokenize(cadena, pattern) #leer el pattern mas arriba
    terminos = ' '.join([str(x) for x in terminos])
    #No se han eliminado repetidos.
    #los eliminamos si queremos ahora
    #terminos = list(set(terminos))
    return terminos

"""creamos un diccionario que relaciona string a id"""
def create_dict(classes):
    res = {}
    for i in range(len(classes)):
        res[classes[i]] = i
    return res

"""Pasamos las clases de string a id integer,
Se le pasa las clases y el diccionario de clases"""
def transform_class_to_integer(y,clases_dict):
    res = []
    #metemos en orden
    for k in y:
        res.append(clases_dict[k])
    return res

def delete_class(X,y,class_to_delete='000'):
    count = 0
    print(len(X))
    print(len(y))
    new_X = []; new_y = []
    for i in range(len(y)):
        if y[i] != class_to_delete:
            new_X.append(X[i])
            new_y.append(y[i])
        else: count += 1
    print("Deleted %d elements " % count)
    return new_X,new_y

def get_sparse_data(perc_train=0.8):
    X, y = read_data()
    X, y = delete_class(X, y)
    clases = list(set(y))
    print("Clases: ", clases)
    n_classes = len(clases)
    print("Numero de clases %d " % n_classes)
    dict_clases = create_dict(clases)

    X, y = shuffle(X, y)

    y = transform_class_to_integer(y, dict_clases)
    print(len(y))
    print("Counter %s " % Counter(y))

    #### -------------- Transforms
    # bag of words - sparse
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X)
    print("After Count Vectorized ")
    print(X_train_counts.shape)
    # print( count_vect.vocabulary_.get(u'aumentar'))

    # tf-idf
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print("After tf ")
    print(X_train_tfidf.shape)

    #### end
    num_train = int(X_train_tfidf.shape[0] * 0.8)
    print("Num ejemplos para entrenar %d " % num_train)
    X_train = X_train_tfidf[:num_train]
    y_train = y[:num_train]
    X_test = X_train_tfidf[num_train:]
    y_test = y[num_train:]

    return X_train, y_train, X_test, X_train


