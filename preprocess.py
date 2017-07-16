import os,csv
import numpy as np
from nltk.tokenize import RegexpTokenizer
import nltk, textblob
from textblob.classifiers import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
from collections import Counter

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
        print(i)
        print(y[i])
        if y[i] != class_to_delete:
            new_X.append(X[i])
            new_y.append(y[i])
        else: count += 1
    print("Deleted %d elements " % count)
    return new_X,new_y

X, y = read_data()
X,y = delete_class(X,y)
clases = list(set(y))
print("Clases: ", clases)
n_classes = len(clases)
print("Numero de clases %d " % n_classes)
dict_clases = create_dict(clases)

y = transform_class_to_integer(y,dict_clases)
print(len(y))
print("Counter %s " % Counter(y))

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=42)
print(np.array(X_train).shape)
print(np.array(y_train).shape)
print(np.array(X_test).shape)
print(np.array(y_test).shape)

train = []
test = []
for i,x in enumerate(zip(X_train,y_train)):
    train.append(x)

for i,x in enumerate(zip(X_train,y_train)):
    test.append(x)
cl = NaiveBayesClassifier(train)
print(cl)

print("Accuracy: {0}".format(cl.accuracy(test)))
