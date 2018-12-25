from sklearn.linear_model import LogisticRegression
from featureProcess import *
from data import *
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score
from confusionMetrics import *
from sklearn.model_selection import GridSearchCV

class logisticModel(LogisticRegression):
    def __init__(self):
        pass
        # super().__init__()
        # self._train_data = AntiFraudData().retrieve_no_missing_value_data(TRAINING_PATH)
        # self._test_data = AntiFraudData().retrieve_no_missing_value_data(TESTING_PATH)
        #

    def best_c_logistic_regression(self, train_X, train_y, test_X, test_y, c_list=np.arange(0.1, 1, 0.1), penalty='l2'):

        auc = []
        for c in c_list:
            print(c)
            # all_solvers = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
            LR = LogisticRegression(C=c, penalty=penalty, solver='liblinear').fit(np.mat(train_X), np.ravel(train_y))
            pred = LR.predict_proba(np.mat(np.mat(test_X)))[:, 1]
            test_auc = roc_auc_score(test_y, pred)
            auc.append(test_auc)
        position = np.argmax(auc)
        c_best = c_list[position]
        print('max auc: ', max(auc))
        LR = LogisticRegression(C=c_best, penalty=penalty, solver='liblinear').fit(np.mat(train_X), np.ravel(train_y))
        # parameters = {'C': c_list}
        # lr = GridSearchCV(n_jobs=-1, estimator=LogisticRegression(penalty=penalty), param_grid=parameters, scoring='f1', cv=5,)
        # LR.fit(train_X, train_y)
        # best_c,

        return LR


if __name__ == '__main__':
    from data import *

    data_units = DataGenerator().get_logistic_regression_data()
    train_X = data_units['train_X']
    train_y = data_units['train_y']
    test_X = data_units['test_X']
    test_y = data_units['test_y']
    lr = logisticModel().best_c_logistic_regression(train_X, train_y, test_X, test_y, )

    df = pd.concat([test_y, pd.DataFrame(lr.predict(test_X), columns=['y'], index=test_X.index)], axis=1)

    ROC_AUC(target='flag', score='y', df=df, plot=True)

    # print(get_confusion_matrix(TP, FP, TN, FN))

