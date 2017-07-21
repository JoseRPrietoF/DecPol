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
from sklearn.feature_extraction.text import CountVectorizer


class Data():

    """
    path: ruta donde estan los csv
    stem: hacer stemming o no
    simly: reducir el numero de clases de 40 a 4
    no_stop_word: quitar stop_words en espanyol y catalan (por defecto, se pueden anyadir o eliminar el catalan)
    delete_class: Por defecto se eliminan las clases '000' y '0', se puede modificar
    codif: 
        - tfidf: Se codifica en tfidf las X
        - freq: se codifican por frecuencia las palabras, cada palabra será un numero
        - bagofwords: se aplica bag of words a las X, cada palabra tambien sera un numero [DEFECTO]
    max_features: se utiliza solo en bag of words. Indica el numero de palabras (tokens) a escoger por orden (TOP). Con None se escogen todas. Default None
    """
    def __init__(self,path="data/", stem=True, simply=True, stop_word = True, delete_class=['0','000'],codif = 'bagofwords',max_features=None):

        self.path = path
        self.stem = stem
        self.simply = simply
        self.stop_words = stop_word
        self.class_to_delete = delete_class
        self.codif = codif
        self.max_features = max_features

        self.tam_voc = -1

        self.lista_f_stopwords = ['catalan.txt']

        self.pattern = r'\w+'

        self.vocab_freq = {}
        self.coding = {}

        self.lista_f_stopwords = []

        self.y = []
        self.X = []

        self.start()

    def start(self):
        self.read_data()
        self.delete_class()

        self.transform_class_to_integer()

        print("Codif : %s " % self.codif)
        if (self.codif == 'tfidf'):
            self.X = self.tf_idf(self.X)

        elif (self.codif == 'freq'):

            items = self.vocab_freq.items()
            ordered = sorted(items, key=itemgetter(1), reverse=True)

            tamanyo = self.create_coding(ordered)

            self.codificar()
            print("X: ")
            print(self.X)
            print("*" * 100)
        elif (self.codif == 'bagofwords'):
            print("Creating the bag of words...\n")
            self.bag_of_word()

        print("Finished. \n Classes: %s" % self.y)

    def bag_of_word(self):
        vectorizer = CountVectorizer(analyzer="word", \
                                     tokenizer=None, \
                                     preprocessor=None, \
                                     stop_words=None, \
                                     max_features=self.max_features)

        # fit_transform() does two functions: First, it fits the model
        # and learns the vocabulary; second, it transforms our training data
        # into feature vectors. The input to fit_transform should be a list of
        # strings.

        train_data_features = vectorizer.fit_transform(self.X)

        # Numpy arrays are easy to work with, so convert the result to an
        # array
        self.X = train_data_features.toarray()

        vocab = vectorizer.get_feature_names()
        print("Vocabulario bag of words: \n %s" % vocab)

    """Metodo al que se le pasa un prcentaje (default=0.8) y mezclara y dividira el dataset y lo devolvera
    El porcentaje indica el numero de training
    return: X_train, y_train, X_test, y_test
    """
    def train_test_split(self,perc_train=0.8):
        self.X, self.y = shuffle(self.X, self.y)
        num_train = int(len(self.X) * perc_train)
        print("Num ejemplos para entrenar %d " % num_train)
        X_train = self.X[:num_train]
        y_train = self.y[:num_train]
        X_test = self.X[num_train:]
        y_test = self.y[num_train:]
        return X_train, y_train, X_test, y_test

    def set_path(self,path):
        self.path = path

    def switch_stemming(self):
        self.stem = not self.stem
        print("Stemming %s" % self.stem)

    """Simplifica las clases, de 40 a 4"""
    def switch_simply(self):
        self.simply = not self.simply
        print("simply %s" % self.simply)

    def switch_stop_words(self):
        self.stop_words = not self.stop_words
        print("stop_words %s" % self.stop_words)


    def elimina_tildes(self,s):
       return ''.join((c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'))


    def lista_ficheros_stopwords(self, lista = []):
        self.lista_f_stopwords = lista

    def add_fichero_stopword(self, fn):
        self.lista_f_stopwords.append(fn)

    def cargar_stopwords(self):
        pal = []
        for fn in self.lista_f_stopwords:
            with open(fn, "r",encoding='utf-8') as myfile:
                pal = self.elimina_tildes(myfile.read())
                pal = pal.split("\n")
        return pal

    """Devuelve X (muestras) e y(etiquetas)"""
    def read_data(self):
        lst = os.listdir(self.path)
        lst.sort()
        X = []
        y = []
        for file_name in lst:
            if not file_name.endswith(".csv"):
                continue
            filepath = os.path.join(self.path, file_name)
            with open(filepath, newline='', encoding="ISO-8859-1") as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='"')
                for row in spamreader:
                    if row[1] == '': continue
                    X.append(self.procesar(row[1]))
                    y.append(row[2])
        print("frecuencia vocab  %s " % self.vocab_freq)
        self.tam_voc = len(self.vocab_freq.keys())
        print("tam vocab %d " % self.tam_voc)
        if self.simply:
            y = self.simplify_classes(y)
            print("Reducido numero de clases!!!!")
        self.y = y
        self.X = X
        return X, y

    """modifica coding creando un numero (segun su frecuencia) para cada palabra, a partir de la lista ordenada por frecuencia.
    Devuelve el tamanyo del diccionario (vocabul)"""
    def create_coding(self,ordered):
        # ('europe', 560), ('per', 335), ('social', 315), ....
        index = 0
        for x, y in ordered:
            self.coding[x] = index
            index += 1
        print(self.coding)
        return index

    """Devuelve los terminos"""
    def procesar(self,cadena):

        #pasamos to do a minus
        cadena = cadena.lower()
        # Quitamos simbolos no alfanumeicos
        terminos = nltk.regexp_tokenize(cadena, self.pattern) #leer el pattern mas arriba

        # This is the simple way to remove stop words
        #Tambien elimina caracteres
        if self.stop_words:
            extra_stop = self.cargar_stopwords()
            terminos_non_stop = []
            for word in terminos:
                if word not in stopwords.words('spanish') and len(word) > 2 and word not in extra_stop:
                    word = self.elimina_tildes(word)
                    word = re.sub(r'[^a-zA-Z ]', '', word) #eliminamos caracteres alfanumericos
                    if word != '':
                        terminos_non_stop.append(re.sub('\W+','', word))
            terminos = terminos_non_stop
        #No se han eliminado repetidos.
        #los eliminamos si queremos ahora
        #terminos = list(set(terminos))
        if self.stem:
            stemmer = SnowballStemmer("spanish")
            for i in range(len(terminos)):
                terminos[i] = stemmer.stem(terminos[i])

        self.add_to_dict(terminos)

        terminos = ' '.join([str(x) for x in terminos])
        return terminos

    """Va creando el diccionario con las frecuencias
    Modifica vocab_freq"""
    def add_to_dict(self,terminos):
        for t in terminos:
            freq = self.vocab_freq.get(t,0)
            self.vocab_freq[t] = freq + 1

    def delete_class(self):
        count = 0
        new_X = [];
        new_y = []
        for i in range(len(self.y)):
            if self.y[i] not in self.class_to_delete:
                new_X.append(self.X[i])
                new_y.append(self.y[i])
            else:
                count += 1
        print("Deleted %d elements " % count)
        self.y = new_y
        self.X = new_X

    """Pasamos las clases de string a id integer,
    Se le pasa las clases y el diccionario de clases"""
    def transform_class_to_integer(self):

        def create_dict(classes):
            res = {}
            for i in range(len(classes)):
                res[classes[i]] = i
            return res

        res = []
        clases = list(set(self.y))
        n_classes = len(clases)
        print("Clases %s " % clases)
        print("Numero de clases %d " % n_classes)
        dict_clases = create_dict(clases)

        # metemos en orden
        for k in self.y:
            res.append(dict_clases[k])
        self.y = res

    def tf_idf(self,X):
        #### -------------- Transforms
        # bag of words - sparse
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(X)

        # print( count_vect.vocabulary_.get(u'aumentar'))

        # tf-idf
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

        self.X = X_train_tfidf

    """Metodo que pasa de la clase
    A202 a A
    BXXXXX a B
    etc
    """
    def simplify_classes(self,y):
        for i in range(len(y)):
            y[i] = y[i][0]
        return y

    """Codifica cada palabra segun el coding
    X = ["hola sdf", "gehh th rthrt h" ...] """
    def codificar(self):
        new_X = []
        for frase in self.X:
            new_frase = []
            frase = frase.split(" ")
            for pal in frase:
                new_frase.append(self.coding.get(pal))
            new_X.append(new_frase)
        self.X = new_X
