import numpy as np

def split_sets(df, train_size, val_size, test_size, target):
    """
    Split the data into train, validation and test sets with target and features
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
    """
    train_size = int(train_size * len(df))
    val_size = int(val_size * len(df))
    test_size = int(test_size * len(df))
    return df[:train_size], df[train_size:train_size+val_size], df[train_size+val_size:train_size+val_size+test_size]

def euclidean_v(x, y):
    return np.sqrt(np.sum((x - y)**2))


def manhattan_v(x, y):
    return np.sum(np.abs(x - y))


def cosine_v(x, y):
    return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
