from data import *
from gradientDescent import *
from confusionMetrics import *
import matplotlib.pyplot as plt
from featureProcess import *
from sklearn import model_selection
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score
from scipy import interp


class DataGenerator:
    def __init__(self, raw_data_path=RAW_DATA_PATH):
        self._raw_data = AntiFraudData(raw_data_path).raw

    def get_logistic_regression_data(self, test_size=0.3):
        """

        :param raw_data_path:
        :param test_size:
        :return:
        """

        train_data, test_data = model_selection.train_test_split(self._raw_data, test_size=test_size)
        print('overall fraud rate on train set is {0:2.4f}%'.format(train_data['flag'].mean() * 100))
        feature_process = BasicFeatureProcess()
        """
        check constant variable both categorical & numerical
        """
        fixed_cols = feature_process.fix_value_check(train_data)
        if fixed_cols:
            print('Constant columns:', fixed_cols)
            train_data = train_data.drop(fixed_cols, axis=1).copy()
        else:
            print('No constant columns')
        train_data.loc[train_data.age == 0, 'age'] = np.nan
        """
        check features' missing percentage
        age ==0 indicates a missing value
        """
        missing_rate_df = feature_process.missing_value_pct(train_data)
        columns_with_missing = missing_rate_df.index[missing_rate_df.missing_rate > 0].tolist()
        """
        add missing indicator
        """
        train_data = feature_process.add_missing_indicator_to_df(data=train_data, cols=columns_with_missing)
        missing_odds_ratio = feature_process.missing_odds_ratio(train_data, cols=columns_with_missing, flag_col='flag')
        print(missing_odds_ratio.head(5))
        """
        check categorical feature
        """
        obj_types = feature_process.feature_type_identifier(train_data, 'O')
        data_counts = train_data.nunique()
        categorical_cols = data_counts.index[data_counts <= 10].tolist()
        categorical_cols = list(set(categorical_cols + obj_types) - set(['flag']))
        """
        for categorical feature with missing value encode with dummy variable and delete original ferature
        """
        categoriacal_cols_with_missing = list(set(columns_with_missing) & set(categorical_cols))
        dummy_result = feature_process.get_categorical_dummy_variable(train_data, categorical_cols)
        train_data = dummy_result['data']
        dummy_columns = dummy_result['dummy_cols']
        """
        continous variable fill missing value as 0, and using missing indicator
        """
        continues_col_with_missing = list(set(columns_with_missing) - set(categoriacal_cols_with_missing))
        train_data[continues_col_with_missing] = train_data[continues_col_with_missing].fillna(0)
        """
        feature derivative: average payment during periods
        """
        feature_derive = FeatureDerivative()
        train_data = feature_derive.avg_payment_derivative(train_data)
        """
        outlier detection & normalization
        """
        all_columns = list(train_data.columns)
        all_columns.remove('ID')
        all_columns.remove('flag')
        numerical_columns = list(set(all_columns) - set(dummy_columns) - set(categorical_cols))
        outlier_columns = get_outlier_features(train_data, numerical_columns, expand=True)
        outlier_fraud_df = outlier_effect(train_data, outlier_cols=outlier_columns, flag_col='flag', expand=True)
        outlier_fraud_df = outlier_fraud_df.fillna(0).replace([-np.inf, np.inf], 0)
        print('outlier_fraud odds ratio')
        print(outlier_fraud_df.loc[(outlier_fraud_df != 0).any(axis=1)])

        # no significant difference
        # feature standarlization
        lower, upper, mu, sigma = {}, {}, {}, {}
        for col in outlier_columns:
            temp_df = train_data[[col, 'flag']]
            zero_score = zero_score_normalization(temp_df, col)
            if zero_score == 1:
                del train_data[col]
                outlier_columns.remove(col)
                numerical_columns.remove(col)
                continue
            train_data[col] = zero_score['new_var']
            lower[col], upper[col], mu[col], sigma[col] = zero_score['lower'], zero_score['upper'], zero_score['mu'], \
                                                          zero_score['sigma']
        # test data
        print('overall fraud rate on test set is {0:2.4f}%'.format(test_data['flag'].mean() * 100))
        test_data = test_data.drop(fixed_cols, axis=1).copy()
        test_data.loc[test_data.age == 0, 'age'] = np.nan
        test_data = feature_process.add_missing_indicator_to_df(data=test_data, cols=columns_with_missing)

        dummy_result = feature_process.get_categorical_dummy_variable(test_data, categorical_cols)
        test_data = dummy_result['data']
        test_data[continues_col_with_missing] = test_data[continues_col_with_missing].fillna(0)
        test_data = feature_derive.avg_payment_derivative(test_data)
        for col in outlier_columns:
            temp_df = test_data[[col, 'flag']]
            if col not in numerical_columns:
                del test_data[col]
                continue
            test_data[col] = (temp_df[col] - mu[col]) / sigma[col]

        train_X = train_data[all_columns]
        train_y = train_data[['flag']]
        match_df = pd.DataFrame(columns=train_data.columns)
        test_data = pd.concat([match_df, test_data],axis=0, sort=True).fillna(0)
        test_X = test_data[all_columns]
        test_y = test_data[['flag']]
        return {'train_X': train_X, 'train_y': train_y, 'test_X': test_X, 'test_y': test_y}

    def get_decision_tree_data(self, raw_data_path=AntiFraudData, test_size=0.3):
        pass


if __name__ == '__main__':
    dg = DataGenerator()
    x, y = dg.get_logit_regression_data()
    print(x)
