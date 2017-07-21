from sklearn.neighbors import KNeighborsClassifier
from preprocess import Data

dataset = Data(path="data/", stem=True, simply=True, stop_word = True, delete_class=['0','000'],codif = 'bagofwords',max_features=None)
X_train, y_train, X_test, y_test = dataset.train_test_split(0.8)

best_acc = 0
best_n = 0

for i in range(1,40,1):

    classifier = KNeighborsClassifier( n_neighbors=i, weights='uniform' )
    print("Fitting KNN Classifier")
    classifier.fit( X_train, y_train )
    print("Predicting with KNN Casiffier")
    y_pred = classifier.predict( X_test )

    print("-"*50)
    print("KNN ")
    print("Numero de vecinos %d " % i)
    acc = ( ( 100.0 * (y_test == y_pred).sum() ) / len(y_test) )
    print( "%d muestras mal clasificadas de %d" % ( (y_test != y_pred).sum(), len(y_test) ) )
    print( "Accuracy = %.1f%%" % acc )
    print("-"*50)
    if best_acc < acc:
        best_acc = acc
        best_n = i

print("*"*100)
print("Mejor KNN con %d vecinos con accuracy %f " % (best_n,best_acc))

"""
****************************************************************************************************
Mejor KNN con 3 vecinos con accuracy 41.747573 
"""