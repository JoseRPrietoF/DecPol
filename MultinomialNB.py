from sklearn.naive_bayes import MultinomialNB
import preprocess
from preprocess import Data


best_acc = 0
best_n = 0
dataset = Data(path="data/", stem=True, simply=True, stop_word = True, delete_class=['0','000'],codif = 'bagofwords',max_features=None)
X_train, y_train, X_test, y_test = dataset.train_test_split(0.8)
for i in [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]:

    clf = MultinomialNB(alpha=i).fit(X_train, y_train)
    acc = clf.score(X_test,y_test)
    #print("****" * 100)
    #print("with %d features " % i)
    #print("Accuracy %f " % acc)
    #print("Vocab %s " % vocab)
    print("Acc: %f" % acc)
    if acc > best_acc:
        print("Acc %f > %f" % (acc,best_acc))
        best_acc = acc
        best_n = i

print("*" * 100)
print("Mejor 'alpha' %f con accuracy %f " % (best_n, best_acc))

"""
Mejor 'alpha' 0.100000 con accuracy 0.757282 
"""