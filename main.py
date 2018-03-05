import pandas as pd
import numpy as np 

def load_data(path):
    users = pd.read_csv(path+'/users.tsv', sep='\t')
    test_users = pd.read_csv(path+'/test_users.tsv', sep='\t')
    datamap = {
        'users':users,
        'test_users':test_users
    }
    return datamap

def data_info(data):
    print 'number of rows:'
    print len(data.index)
    print 'number of columns:'
    print len(data.columns)


if __name__ == '__main__':
    datamap = load_data('./dataset')
    users = datamap['users']
    data_info(users)