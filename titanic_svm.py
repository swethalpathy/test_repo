from sklearn.svm import SVC

# SVM
def svm(xtrain, ytrain, xval):
    global svm, svm, y_pred_svm
    svm = SVC(kernel='rbf', random_state=20, probability=True)
    model_svm = svm.fit(xtrain, ytrain)
    y_pred_svm = svm.predict(xval)