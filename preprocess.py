import os,csv
from nltk.tokenize import RegexpTokenizer
import nltk, re
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from operator import itemgetter
import unicodedata

pattern = r'\w+'
tokenizerAlpha = RegexpTokenizer(pattern)

vocab_freq = {}
coding = {}


def elimina_tildes(s):
   return ''.join((c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'))

def cargar_stopwords_catalan():
    pal = []
    with open("catalan.txt", "r",encoding='utf-8') as myfile:
        pal = elimina_tildes(myfile.read())
        pal = pal.split("\n")
    return pal



"""modifica coding creando un numero (segun su frecuencia) para cada palabra, a partir de la lista ordenada por frecuencia.
Devuelve el tamanyo del diccionario (vocabul)"""
def create_coding(ordered):
    #('europe', 560), ('per', 335), ('social', 315), ....
    index = 0
    for x,y in ordered:
        coding[x] = index
        index += 1
    print(coding)
    return index

"""Devuelve los terminos"""
def procesar(cadena,stem = True, extra_stop = []):

    #pasamos to do a minus
    cadena = cadena.lower()
    # Quitamos simbolos no alfanumeicos
    terminos = nltk.regexp_tokenize(cadena, pattern) #leer el pattern mas arriba

    # This is the simple way to remove stop words
    #Tambien elimina caracteres
    terminos_non_stop = []
    for word in terminos:
        if word not in stopwords.words('spanish') and len(word) > 2 and word not in extra_stop:
            word = elimina_tildes(word)
            word = re.sub(r'[^a-zA-Z ]', '', word) #eliminamos caracteres alfanumericos
            if word != '':
                terminos_non_stop.append(re.sub('\W+','', word))
    terminos = terminos_non_stop
    #No se han eliminado repetidos.
    #los eliminamos si queremos ahora
    #terminos = list(set(terminos))
    if stem:
        stemmer = SnowballStemmer("spanish")
        for i in range(len(terminos)):
            terminos[i] = stemmer.stem(terminos[i])

    add_to_dict(terminos)

    terminos = ' '.join([str(x) for x in terminos])
    return terminos

"""Va creando el diccionario con las frecuencias
Modifica vocab_freq"""
def add_to_dict(terminos):
    for t in terminos:
        freq = vocab_freq.get(t,0)
        vocab_freq[t] = freq + 1

"""creamos un diccionario que relaciona string a id"""
def create_dict(classes):
    res = {}
    for i in range(len(classes)):
        res[classes[i]] = i
    return res

"""Pasamos las clases de string a id integer,
Se le pasa las clases y el diccionario de clases"""
def transform_class_to_integer(y):
    res = []
    clases = list(set(y))
    n_classes = len(clases)
    print("Clases %s " % clases)
    print("Numero de clases %d " % n_classes)
    dict_clases = create_dict(clases)

    #metemos en orden
    for k in y:
        res.append(dict_clases[k])
    return res

def delete_class(X,y,class_to_delete=['000','0']):
    count = 0
    new_X = []; new_y = []
    for i in range(len(y)):
        if y[i] not in class_to_delete:
            new_X.append(X[i])
            new_y.append(y[i])
        else: count += 1
    print("Deleted %d elements " % count)
    return new_X,new_y

def tf_idf(X):
    #### -------------- Transforms
    # bag of words - sparse
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X)

    # print( count_vect.vocabulary_.get(u'aumentar'))

    # tf-idf
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    return X_train_tfidf

"""Metodo que pasa de la clase
A202 a A
BXXXXX a B
etc
"""
def simplify_classes(y):
    for i in range(len(y)):
        y[i] = y[i][0]
    return y

"""Codifica cada palabra segun el coding
X = ["hola sdf", "gehh th rthrt h" ...] """
def codificar(X,coding={}):
    new_X = []
    for frase in X:
        new_frase = []
        frase = frase.split(" ")
        for pal in frase:
            new_frase.append(coding.get(pal))
        new_X.append(new_frase)
    return new_X

def get_data(perc_train=0.8, codif = 'tfidf', stem = True, simply=False):
    X, y = read_data(stem=stem,simply=simply)
    X, y = delete_class(X, y)

    X, y = shuffle(X, y)

    y = transform_class_to_integer(y)

    print("Codif : %s " % codif)
    if (codif == 'tfidf'):
        X = tf_idf(X)
        X_train, y_train, X_test, y_test = train_test_split(perc_train)

        # tamanyo = n_features
        return X_train, y_train, X_test, y_test
    elif (codif == 'freq'):


        items = vocab_freq.items()
        ordered = sorted(items, key=itemgetter(1), reverse=True)

        tamanyo = create_coding(ordered)

        X = codificar(X, coding)
        print("X: ")
        print(X)
        print("*"*100)
        X_train, y_train, X_test, y_test = train_test_split(perc_train)
        print("Tamanyo del dic %d " % tamanyo)
        return X_train, y_train, X_test, y_test, tamanyo, coding
    elif (codif == 'bagofwords'):
        return X,y


def train_test_split(X,y,perc_train):
    X, y = shuffle(X, y)
    num_train = int(len(X) * perc_train)
    print("Num ejemplos para entrenar %d " % num_train)
    X_train = X[:num_train]
    y_train = y[:num_train]
    X_test = X[num_train:]
    y_test = y[num_train:]
    return X_train, y_train, X_test, y_test

"""Devuelve X (muestras) e y(etiquetas), tamanyo_dic, codificacion"""
def read_data(path="data/", stem=True,simply = False):
    stop_catalan = cargar_stopwords_catalan()

    lst = os.listdir(path)
    lst.sort()
    X = []
    y = []
    for file_name in lst:
        if not file_name.endswith(".csv"):
            continue
        filepath = os.path.join(path, file_name)
        with open(filepath, newline='',encoding="ISO-8859-1") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in spamreader:
                if row[1] == '': continue
                X.append(procesar(row[1], stem,extra_stop=stop_catalan))
                y.append(row[2])
    print("frecuencia vocab  %s " % vocab_freq)
    print("tam vocab %d " % len(vocab_freq.keys()))
    if simply:
        y = simplify_classes(y)
        print("Reducido numero de clases!!!!")

    return X,y



#get_data(simply=True)