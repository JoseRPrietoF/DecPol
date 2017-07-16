from sklearn.naive_bayes import MultinomialNB
import preprocess

X_train, y_train, X_test, y_test = preprocess.get_sparse_data()


clf = MultinomialNB().fit(X_train, y_train)

print(clf.score(X_test,y_test))