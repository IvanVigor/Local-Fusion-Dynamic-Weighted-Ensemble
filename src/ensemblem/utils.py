import numpy as np

def split_sets(df, train_size, val_size, test_size, target):
    """
    Split the data into train, validation and test sets with target and features

    Parameters
    ----------

    :param df: pandas.DataFrame to be divided
    :param train_size: float, size of the train set
    :param val_size: float, size of the validation set
    :param test_size: float, size of the test set

    :return: train, validation and test sets with target and features

    """
    train, val, test = divide_sets(df, train_size, val_size, test_size)
    return train.drop(columns=[target]), \
           train[target], \
           val.drop(columns=[target]), \
           val[target], \
           test.drop(columns=[target]), \
           test[target]

def divide_sets(df, train_size, val_size, test_size):
    """
    Divide the data into train, validation and test sets

    Parameters
    ----------
    :param df: pandas.DataFrame to be divided
    :param train_size: float, size of the train set
    :param val_size: float, size of the validation set
    :param test_size: float, size of the test set

    :return: train, validation and test sets
    """
    train_size = int(train_size * len(df))
    val_size = int(val_size * len(df))
    test_size = int(test_size * len(df))
    return df[:train_size], df[train_size:train_size+val_size], df[train_size+val_size:train_size+val_size+test_size]
