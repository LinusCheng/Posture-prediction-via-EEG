import util_EEG as utl
import run_model as m
import numpy as np


""" RNN """
#SET 1
# Bidirectional avg pooling stride 15
test_set   = 1
data_list = utl.load_dataset_sim(test_set)
m.bi_LSTM(data_list)



""" ConvNet_Nocropping Each Set """
#SET 1
test_set   = 1
data_list = utl.load_dataset_sim(test_set)
m.ConvNet_nocrop(data_list)




""" ConvNet Each Set """
#SET 1
test_set   = 1
data_list = utl.load_dataset(test_set)
m.ConvNet(data_list,epoch_all=60,batch_size=None,lr_decay=0.975)


""" ConvNet 1 vs 8 """
#SET 1 vs all others
test_set   = 1
batch_size = 390
data_list = utl.load_1vs8_data(test_set)
m.ConvNet(data_list,epoch_all=60,batch_size=batch_size,lr_decay=0.975)



""" ConvNet mix """
test_data_id = np.arange(2558)
np.random.shuffle(test_data_id)
test_data_id=test_data_id[:300]
print('test_data_id:')
print(test_data_id)
print("\n")
batch_size = 370
data_list  = utl.load_mix_data(test_data_id)
m.ConvNet(data_list,epoch_all=60,batch_size=batch_size,lr_decay=0.975)



