from sklearn.linear_model import LogisticRegression
from data import *
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score
from confusionMetrics import *


class logisticModel(LogisticRegression):
    def __init__(self):
        super().__init__()
        # self._train_data = AntiFraudData().retrieve_no_missing_value_data(TRAINING_PATH)
        # self._test_data = AntiFraudData().retrieve_no_missing_value_data(TESTING_PATH)
        #

    def best_c_logistic_regression(self, train_X, train_y, test_X, test_y, c_list=np.arange(0.01, 1, 0.01), penalty='l2'):

        auc = []
        for c in c_list:
            LR = LogisticRegression(C=c, penalty=penalty).fit(train_X, train_y)
            pred = LR.predict_proba(test_X)[:, 1]
            test_auc = roc_auc_score(test_y, pred)
            auc.append(test_auc)
        position = np.argmax(auc)
        c_best = c_list[position]
        print(max(auc))
        LR = LogisticRegression(C=c_best).fit(train_X, train_y)
        return LR

