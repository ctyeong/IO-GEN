import numpy as np
from PIL import Image
from glob import glob
import os
import pandas as pd


def load_of_data(split_dir, m, max_m=4):

    trains = pd.read_csv(os.path.join(split_dir, 'train.csv'), names=['paths'])
    tests = pd.read_csv(os.path.join(split_dir, 'test.csv'), names=['paths'])

    train_x = []
    test_stable_x = []
    test_unstable_x = []
    
    # train
    for name in trains['paths']: 
        x = []
        for i in range(m):
            folder, number = name.split('/')
            flow_x = folder + '/flow_x_' + number + '-{}.jpg'.format(i)
            flow_y = folder + '/flow_y_' + number + '-{}.jpg'.format(i)
            x.append(np.asarray(Image.open(flow_x).convert('L'))/255.*2.-1.)
            x.append(np.asarray(Image.open(flow_y).convert('L'))/255.*2.-1.)
        train_x.append(x)
    train_x = np.transpose(np.array(train_x), (0,2,3,1))
    
    # test_stable
    for name in tests['paths']: 
        x = []
        for i in range(m):
            folder, number = name.split('/')
            flow_x = folder + '/flow_x_' + number + '-{}.jpg'.format(i)
            flow_y = folder + '/flow_y_' + number + '-{}.jpg'.format(i)
            x.append(np.asarray(Image.open(flow_x).convert('L'))/255.*2.-1.)
            x.append(np.asarray(Image.open(flow_y).convert('L'))/255.*2.-1.)
        test_stable_x.append(x)
    test_stable_x = np.transpose(np.array(test_stable_x), (0,2,3,1))

    # test_unstable 
    flow_x_list = sorted(glob('./Unstable/flow_x_*.jpg'))
    flow_y_list = sorted(glob('./Unstable/flow_y_*.jpg'))
    
    for i in range(len(flow_x_list)//max_m):
        x = []  
        for j in range(m): 
            x.append(np.asarray(Image.open(flow_x).convert('L'))/255.*2.-1.)
            x.append(np.asarray(Image.open(flow_y).convert('L'))/255.*2.-1.)
        test_unstable_x.append(x)
    test_unstable_x = np.transpose(np.array(test_unstable_x), (0,2,3,1))

    return train_x, test_stable_x, test_unstable_x

