from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from nltk.stem import SnowballStemmer
import preprocess
from bagOfWords import bagOfWords,bagOfWords_cargado,bagOfWords_Cargar


stem = False
simply = True
obj = bagOfWords_Cargar(simply=simply,stem=stem)

best_acc = 0
best_n = 0

print("Clases: %s " % obj.get('y'))
for i in [8002]:

    X,y,vocab = bagOfWords_cargado(obj, max_features=i)

    X_train, y_train, X_test, y_test = preprocess.train_test_split(X,y,0.8)

    best_acc = 0
    best_n = 0

    params = {'kernel': ['rbf', 'linear'],
              #'C': [1.0, 10.0, 100.0, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 1.0e+1],
              'C': [0.001,0.01,0.1,1,10,100,200,1000],
              'degree': [1],
              'gamma': [1e-07,1e-06,0.00001,0.0001,0.001, 0.01, 0.1],
              'coef0': [0.0]
              }
    param_grid = {'params': [params]}

    svr = GridSearchCV(SVC(max_iter=1000), params,verbose=True)
    svr.fit(X_train, y_train)
    print('Best regressor for %s ' % svr.best_estimator_)
    acc = svr.score(X_test, y_test)
    print("Acc %f " % acc)
    print("i %d " % i)
    #print("****" * 100)
    #print("with %d features " % i)
    #print("Accuracy %f " % acc)
    #print("Vocab %s " % vocab)
    if acc > best_acc:
        best_acc = acc
        best_n = i

print("*" * 100)
print("Mejor 'max_features' %d con accuracy %f " % (best_n, best_acc))

"""
Best regressor for SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=1, gamma=0.001, kernel='rbf',
  max_iter=1000, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False) 
Acc 0.786408 

Best regressor for SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=1, gamma=1e-05, kernel='linear',
  max_iter=1000, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False) 
Acc 0.776699

Best regressor for SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=1, gamma=1e-07, kernel='linear',
  max_iter=1000, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False) 
Acc 0.786408 

Best regressor for SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=1, gamma=1e-07, kernel='linear',
  max_iter=1000, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False) 
Acc 0.786408 
i 6300 


Fitting 3 folds for each of 112 candidates, totalling 336 fits
[Parallel(n_jobs=1)]: Done 336 out of 336 | elapsed:  7.4min finished
Best regressor for SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=1, gamma=1e-07, kernel='linear',
  max_iter=1000, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False) 
Acc 0.815534 
i 6400 
"""