from sklearn.ensemble import RandomForestClassifier

# Random forest
def randomForest(xtrain, ytrain, xval):
    global rf, y_pred_rf, rf
    rf = RandomForestClassifier(bootstrap=True,
                                criterion='entropy',
                                max_depth=None,
                                max_features='auto',
                                min_samples_leaf=1,
                                min_samples_split=10,
                                n_estimators=100)
    model_rf = rf.fit(xtrain, ytrain)
    y_pred_rf = rf.predict(xval)