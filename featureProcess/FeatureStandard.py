import numpy as np
import pandas as pd


def mean_std(array):
    return np.mean(array), np.std(array)


def linear_std(array):
    min = np.min(array)
    max = np.max(array)
    return (array - min) / (max - min)


def norm_std(array):
    mean = np.mean(array)
    std = np.std(array)
    return (array - mean) / std


def get_outlier_features(data, feature_list, percentiles=(0.1, 0.99),  expand=False):
    """

    :param data:
    :param feature_list:
    :param percentiles:
    :return:
    """
    df = data.copy()
    lower = percentiles[0]
    upper = percentiles[1]
    quantile_df = df[feature_list].quantile([lower, upper], numeric_only=False)
    outlier_feature = list()
    for feature in feature_list:
        l = quantile_df[feature][lower]
        u = quantile_df[feature][upper]
        if expand:
            d = u - l
            u, l = u + 1.5 * d, l - 1.5 * d
        out_counts = sum((df[feature]<l)|(u<df[feature]))
        if out_counts:
            outlier_feature.append(feature)
    return outlier_feature


def outlier_effect(data, outlier_cols, flag_col, percentiles=(1, 99), expand=False):
    """

    :param expand:
    :param data:
    :param outlier_cols:
    :param flag_col:
    :param percentiles:
    :return:
    """

    outlier_effect_dict = dict()
    df = data.copy()
    for col in outlier_cols:
        lower, upper = np.percentile(df[col], percentiles[0]), np.percentile(df[col], percentiles[1])
        if expand:
            d = upper - lower
            upper, lower = upper + 1.5 * d, lower - 1.5 * d
        lower_sample, middle_sample, upper_sample = df[df[col] < lower], df[(df[col] >= lower) & (df[col] <= upper)], \
                                                    df[df[col] > upper]
        lower_fraud, middle_fraud, upper_fraud = lower_sample[flag_col].mean(), middle_sample[flag_col].mean(), \
                                                 upper_sample[flag_col].mean()
        l = (lower_fraud / middle_fraud)
        u = (upper_fraud / middle_fraud)
        lower_log_odds, upper_log_odds = np.log(l), np.log(u)
        outlier_effect_dict[col] = {'log_odds_lower': lower_log_odds, 'log_odds_upper': upper_log_odds}
    res_df = pd.DataFrame.from_dict(outlier_effect_dict, orient='index')
    return res_df


def zero_score_normalization(df, col, percentiles=(1, 99), expand=False):
    """

    :param expand:
    :param df:
    :param col:
    :param percentiles:
    :return:
    """
    lower, upper = np.percentile(df[col], percentiles[0]), np.percentile(df[col], percentiles[1])
    if lower == upper:
        return 1
    else:
        if expand:
            d = upper - lower
            upper, lower = upper + 1.5 * d, lower - 1.5 * d
        new_col = df[col].map(lambda x: min(max(x, lower), upper))
        mu, sigma = new_col.mean(), np.sqrt(new_col.var())
        new_var = norm_std(new_col)
        return {'new_var': new_var, 'lower': lower, 'upper': upper, 'mu': mu, 'sigma': sigma}


if __name__ == '__main__':
    # （1）生成一个长度1000的序列a，元素是0到10之间的均匀分布的随机数
    a = np.random.uniform(0, 10, 1000)
    print(a)

    # （2）生成一个长度5的序列b，元素是50到100之间的均匀分布的随机数
    b = np.random.uniform(50, 100, 5)
    print(b)
    # （3）将a与b合并在一起，形成序列c
    c = np.concatenate((a, b), axis=0)
    print(c)

    # （4）求出序列a和c的平均值和标准差，并做比较
    a_stats = mean_std(a)
    b_stats = mean_std(c)
    print(a_stats, b_stats)

    # （5）求出序列c的第5%和95%的分位点
    c_5 = np.percentile(c, 5)
    c_95 = np.percentile(c, 95)
    print(c_5, c_95)

    # （6）检验b中的元素，是否落在（5）中的两个分位点之间
    for i in b:
        if c_5 <= i <= c_95:
            print(i, ': between 5% and 95%')
        else:
            print(i, ": out of range")

    # （7）c做线性归一化和均值-标准差归一化
    linear_std_c = linear_std(c)
    norm_std_c = norm_std(c)
    print(linear_std_c)
    print(norm_std_c)

    # （8）去掉c不属于5%和95%分位点之间的值，剩下的值计算max，min，，。再对全部样本做线性归一化和均值-标准差归一化
    c_subset = c[(c_5 <= c) & (c <= c_95)]
    print(linear_std(c_subset))
    print(norm_std(c_subset))
