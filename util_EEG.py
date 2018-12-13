import numpy as np
import h5py


import os
if not os.path.exists('model'):
    os.makedirs('model')


def load_eeg(file_name):
    A0T = h5py.File(file_name, 'r')
    xnp = np.copy(A0T['image'])
    ynp = np.copy(A0T['type'])
    ynp = ynp[0,0:xnp.shape[0]:1]
    ynp = np.asarray(ynp, dtype=np.int32)
    ynp -= 769
    
    xnp = np.swapaxes(xnp,1,2)
    xnp = np.delete(xnp,[22,23,24],axis =2)
    xnp = np.expand_dims(xnp, axis=-1)  # 12/5 added

    rm_mask = np.unique(np.argwhere(np.isnan(xnp))[:,0])
    xnp = np.delete(xnp,rm_mask,axis =0)
    ynp = np.delete(ynp,rm_mask,axis =0)
    
    return xnp, ynp

def load_train_test(test_data_id,total_data_num):

    file_name = ['A01T_slice.mat','A02T_slice.mat','A03T_slice.mat','A04T_slice.mat','A05T_slice.mat',
                 'A06T_slice.mat','A07T_slice.mat','A08T_slice.mat','A09T_slice.mat']
    xnp_train = np.zeros((1,1000,22,1))
    ynp_train = np.zeros(1,dtype=int)
    for i in np.arange(total_data_num):

        if i == test_data_id:
            xnp_test, ynp_test = load_eeg(file_name[i])
        else:
            xnp_i, ynp_i = load_eeg(file_name[i])          
            xnp_train = np.concatenate((xnp_train,xnp_i),axis = 0)
            ynp_train = np.concatenate((ynp_train,ynp_i),axis = 0)
    xnp_train = np.delete(xnp_train,0,0)
    ynp_train = np.delete(ynp_train,0,0)
    return xnp_train,ynp_train,xnp_test,ynp_test



def load_train_test_mix(test_data_id):
    file_name = ['A01T_slice.mat','A02T_slice.mat','A03T_slice.mat','A04T_slice.mat','A05T_slice.mat',
                 'A06T_slice.mat','A07T_slice.mat','A08T_slice.mat','A09T_slice.mat']
    for i in np.arange(9):
        if i == 0:
            xnp_train, ynp_train = load_eeg(file_name[i])
        else:
            xnp_i, ynp_i = load_eeg(file_name[i])          
            xnp_train = np.concatenate((xnp_train,xnp_i),axis = 0)
            ynp_train = np.concatenate((ynp_train,ynp_i),axis = 0)
            
    xnp_test = xnp_train[test_data_id]
    ynp_test = ynp_train[test_data_id]
    xnp_train = np.delete(xnp_train,test_data_id,axis=0)
    ynp_train = np.delete(ynp_train,test_data_id,axis=0)
            
    return xnp_train,ynp_train,xnp_test,ynp_test




def one_hot(Y):
    Hot_Y = np.zeros([len(Y),4])
    Hot_Y[np.arange(len(Y)), Y] = 1
    return Hot_Y





""" load data sets"""
def load_dataset_sim(test_set):

    file_name = ['A01T_slice.mat','A02T_slice.mat','A03T_slice.mat','A04T_slice.mat','A05T_slice.mat',
                     'A06T_slice.mat','A07T_slice.mat','A08T_slice.mat','A09T_slice.mat']
    test_set_str = file_name[test_set-1]

    x, y = load_eeg(test_set_str)
    train_len = x.shape[0]-50
    test_len  = 50

    x_train = x[:-50]
    x_test = x[-50:]
    mean = x_train.mean()
    std = x_train.std()
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std


    #change y to one hot
    # y_hot = one_hot(y)
    y_train = y[:-50]
    y_train = one_hot(y_train)

    y_test  = one_hot(y[-50:])

    # y_test  = np.repeat(y_test,467,axis=0)

    print("X",x.shape)
    print("Y",y.shape)
    print("Training trials =",y.shape[0]-50)
    del x, y
    print("----------------------------------------")
    return [train_len, test_len ,x_train, y_train, x_test , y_test ]



def load_dataset(test_set):

    file_name = ['A01T_slice.mat','A02T_slice.mat','A03T_slice.mat','A04T_slice.mat','A05T_slice.mat',
                     'A06T_slice.mat','A07T_slice.mat','A08T_slice.mat','A09T_slice.mat']
    test_set_str = file_name[test_set-1]

    x, y = load_eeg(test_set_str)
    train_len = x.shape[0]-50
    test_len  = 50

    x_train = x[:-50]
    x_test = x[-50:]
    mean = x_train.mean()
    std = x_train.std()
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std


    #change y to one hot
    # y_hot = one_hot(y)
    y_train = y[:-50]
    y_train = one_hot(y_train)

    y_test  = y[-50:]
    y_test_crop = np.repeat(one_hot(y_test),467,axis=0) 

    # y_test  = np.repeat(y_test,467,axis=0)

    print("X",x.shape)
    print("Y",y.shape)
    print("Training trials =",y.shape[0]-50)
    del x, y
    print("----------------------------------------")
    return [train_len, test_len ,x_train, y_train, x_test , y_test , y_test_crop]




def load_mix_data(test_data_id):
    x_train, y_train, x_test , y_test = load_train_test_mix(test_data_id)

    train_len = x_train.shape[0]
    test_len  = x_test.shape[0]

    mean = x_train.mean()
    std = x_train.std()
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    #change y to one hot
    y_train = one_hot(y_train)
    y_test_crop = np.repeat(one_hot(y_test),467,axis=0) 


    print("x_train",x_train.shape)
    print("y_train",y_train.shape)
    print("Training trials =",train_len)
    print("Testing  trials =",test_len)
    print("----------------------------------------")

    return [train_len, test_len ,x_train, y_train, x_test , y_test , y_test_crop]



def load_1vs8_data(test_set):
    x_train, y_train, x_test , y_test = load_train_test(test_set-1,9)

    train_len = x_train.shape[0]
    test_len  = x_test.shape[0]

    mean = x_train.mean()
    std = x_train.std()
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    #change y to one hot
    y_train = one_hot(y_train)
    y_test_crop = np.repeat(one_hot(y_test),467,axis=0) 


    print("x_train",x_train.shape)
    print("y_train",y_train.shape)
    print("Training trials =",train_len)
    print("Testing  trials =",test_len)
    print("----------------------------------------")

    return [train_len, test_len ,x_train, y_train, x_test , y_test , y_test_crop]

