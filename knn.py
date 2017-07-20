from sklearn.neighbors import KNeighborsClassifier
import preprocess

tf_idf = False
X_train, y_train, X_test, y_test,n_features, _ = preprocess.get_data(simply=True, tfidf=tf_idf)
for i in range(2):
    print("***"*10)
    print(X_train[i])

best_acc = 0
best_n = 0
print("****" * 100)
if tf_idf:
    print("Shape")
    print(X_train[0].shape)
    print("****" *100)
else:
    for i in range(5):
        print("*"*10)
        #print(X_train[i])
        #print(len(X_train[i]))
for i in range(4,100):

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