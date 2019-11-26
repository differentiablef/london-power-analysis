from sklearn.svm import SVC
from sklearn.svm import LinearSVC 

from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

from collections import Counter

def polyTrial(param_grid, target_names, X_train, X_test, y_train, y_test):

    model = SVC(kernel = 'poly', max_iter = 1000000)
    grid = GridSearchCV(model, param_grid, verbose=3, n_jobs=-1, cv=5)
    grid.fit(X_train, y_train)
    print(grid.best_params_)

    model = SVC(verbose=3, kernel = 'rbf', max_iter = 1000000, **grid.best_params_)
    model.fit(X_train, y_train)
    print(" .. Done")
    
    predictions = model.predict(X_train)
    print("Train accuracy = ", round(Counter(y_train-predictions)[0] / 
                                     sum(Counter(y_train-predictions).values()),2))

    predictions = model.predict(X_test)
    print("Test accuracy  = ", round(Counter(y_test-predictions)[0] / 
                                     sum(Counter(y_test-predictions).values()),2))

    print(classification_report(y_test, predictions, target_names=target_names))
    pass

def rbfTrial(param_grid, target_names, X_train, X_test, y_train, y_test):
    model = SVC(kernel = 'rbf', 
                max_iter = 1000000)
    
    grid = GridSearchCV(model, param_grid, 
                        verbose=3, n_jobs=-1, cv=5)
    grid.fit(X_train, y_train)
    print(grid.best_params_)

    model = SVC(verbose=3, kernel = 'rbf', max_iter = 1000000, **grid.best_params_)
    model.fit(X_train, y_train)
    print(" .. Done")
    
    predictions = model.predict(X_train)
    print(classification_report(y_test, predictions, target_names=target_names))
    pass

def linearTrial(param_grid, target_names, X_train, X_test, y_train, y_test):
    model = SVC(kernel = 'linear', max_iter = 1000000)
    grid = GridSearchCV(model, param_grid, verbose=3, n_jobs=-1, cv=5)
    grid.fit(X_train, y_train)
    print(grid.best_params_)

    model = SVC(verbose=3, kernel = 'rbf', max_iter = 1000000, **grid.best_params_)
    model.fit(X_train, y_train)
    print(" .. Done")

    predictions = model.predict(X_train)
    print("Train accuracy = ", round(Counter(y_train-predictions)[0] /
                                     sum(Counter(y_train-predictions).values()),2))

    predictions = model.predict(X_test)
    print("Test accuracy  = ", round(Counter(y_test-predictions)[0] /
                                     sum(Counter(y_test-predictions).values()),2))

    print(classification_report(y_test, predictions, target_names=target_names))
    pass

def linearSVCTrial(param_grid, target_names, X_train, X_test, y_train, y_test):
    model = LinearSVC(max_iter = 1000000)
    grid = GridSearchCV(model, param_grid, verbose=3, n_jobs=-1, cv=5)
    grid.fit(X_train, y_train)
    print(grid.best_params_)

    model = SVC(verbose=3, kernel = 'rbf', max_iter = 1000000, **grid.best_params_)
    model.fit(X_train, y_train)
    print(" .. Done")

    predictions = model.predict(X_train)
    print("Train accuracy = ", round(Counter(y_train-predictions)[0] /
                                     sum(Counter(y_train-predictions).values()),2))

    predictions = model.predict(X_test)
    print("Test accuracy  = ", round(Counter(y_test-predictions)[0] /
                                     sum(Counter(y_test-predictions).values()),2))

    print(classification_report(y_test, predictions, target_names=target_names))
    pass
    
