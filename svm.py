from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from nltk.stem import SnowballStemmer
import preprocess

X_train, y_train, X_test, y_test = preprocess.get_data(simply=True,tfidf=False)

params={ 'kernel': [ 'poly', 'rbf', 'linear', 'sigmoid' ],
         'C' : [1.0, 10.0, 100.0, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 1.0e+1 ],
         'degree' : [ 1, 2, 3 ],
         'gamma': [ 0.001, 0.01, 0.1 ],
         'coef0' : [ 0.0 ] }
param_grid={ 'params' : [ params ] }

svr = GridSearchCV( SVR( max_iter=1000 ), params )
svr.fit( X_train, y_train )
print( 'Best regressor for %s ' %  svr.best_estimator_ )

print(svr.score(X_test,y_test))