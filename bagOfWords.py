print("Creating the bag of words...\n")
from sklearn.feature_extraction.text import CountVectorizer
import preprocess

def bagOfWords(simply=False,stem = False,max_features = 1000):

    clean_train_reviews,y = preprocess.read_data(simply = simply,stem=stem)
    clean_train_reviews, y = preprocess.delete_class(clean_train_reviews, y)
    y = preprocess.transform_class_to_integer(y)

    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = max_features)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.

    train_data_features = vectorizer.fit_transform(clean_train_reviews)


    # Numpy arrays are easy to work with, so convert the result to an
    # array
    train_data_features = train_data_features.toarray()

    vocab = vectorizer.get_feature_names()

    return train_data_features,y,vocab

def bagOfWords_Cargar(type='obj',simply=False,stem = False):
    X, y = preprocess.read_data(simply=simply,stem=stem)
    X, y = preprocess.delete_class(X, y)
    y = preprocess.transform_class_to_integer(y)
    if type == 'obj':

        obj = {'X':X,
               'y':y}
        return obj
    else:
        return X, y

def bagOfWords_cargado(obj,max_features = 1000):

    X = obj.get('X')
    y = obj.get('y')
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    vectorizer = CountVectorizer(analyzer="word", \
                                 tokenizer=None, \
                                 preprocessor=None, \
                                 stop_words=None, \
                                 max_features=max_features)

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of
    # strings.

    train_data_features = vectorizer.fit_transform(X)

    # Numpy arrays are easy to work with, so convert the result to an
    # array
    train_data_features = train_data_features.toarray()

    vocab = vectorizer.get_feature_names()

    return train_data_features, y, vocab