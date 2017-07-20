from sklearn.naive_bayes import MultinomialNB
import preprocess
from keras.preprocessing import sequence
from bagOfWords import bagOfWords,bagOfWords_cargado,bagOfWords_Cargar

obj = bagOfWords_Cargar(simply=True)

best_acc = 0
best_n = 0
print("Clases: %s " % obj.get('y'))
for i in range(100,5300,100):

    X,y,vocab = bagOfWords_cargado(obj, max_features=i)
    X_train, y_train, X_test, y_test = preprocess.train_test_split(X,y,0.8)

    best_acc = 0
    best_n = 0

    clf = MultinomialNB().fit(X_train, y_train)
    acc = clf.score(X_test,y_test)
    #print("****" * 100)
    #print("with %d features " % i)
    #print("Accuracy %f " % acc)
    #print("Vocab %s " % vocab)
    if best_acc < acc:
        best_acc = acc
        best_n = i

print("*" * 100)
print("Mejor 'max_features' %d con accuracy %f " % (best_n, best_acc))