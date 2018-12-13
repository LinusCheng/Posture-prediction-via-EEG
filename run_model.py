import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats



def get_batch_id(batchsize,datalen):
    id_all = np.arange(datalen)
    np.random.shuffle(id_all)   
    id_list = []    
    for i in range(int(datalen/batchsize)):
        id_batch = id_all[int(i*batchsize):int(i*batchsize)+batchsize]
        id_list.append(id_batch)        
    if datalen % batchsize !=0:
        i+=1
        id_batch = id_all[int(i*batchsize):]
        id_list.append(id_batch)        
    return id_list

""" RNN """
def bi_LSTM(data_list):

    [train_len, test_len ,x_train, y_train, x_test , y_test] = data_list
    
    #Define graph
    gf = tf.Graph()
    with gf.as_default():
        initK = tf.contrib.layers.xavier_initializer()
        #Placeholders
        X  = tf.placeholder(dtype=tf.float32,shape=(None,1000,22,1),name='X')
        Y  = tf.placeholder(dtype=tf.float32,shape=None,name='Y')
        T  = tf.placeholder(tf.bool)
        LR = tf.placeholder(tf.float32)


        H = tf.layers.conv2d(X,filters=40,kernel_size=(25,1),strides=(1, 1),padding='valid',activation=None ,kernel_initializer=initK)
        H = tf.layers.batch_normalization(H,momentum=0.1,training=T) 
        # dim N*976*22*40

        H = tf.layers.conv2d(H,filters=40,kernel_size=(1,22),strides=(1, 1),padding='valid',activation=None ,kernel_initializer=initK)
        H = tf.layers.batch_normalization(H,momentum=0.1,training=T) 
        # dim N*976*1*40

        H = tf.square(H)
        
        ## Avg pooling ##
        
#         H = tf.layers.average_pooling2d(H , pool_size=(75,1) , strides=(1, 1), padding='valid')
#         # dim N*902*1*40  like a median filter 902 may be too long
#         H = tf.reshape(H,[-1,902,40])
        # dim N*902*40  
        
        H = tf.layers.average_pooling2d(H , pool_size=(75,1) , strides=(15, 1), padding='valid')
        # dim N*61*1*40      
        H = tf.reshape(H,[-1,61,40])
        # dim N*61*40  
        
        H = tf.log(H)
        H = tf.layers.dropout(H,rate=0.5,training=T)

        ## LSTM ##
        Cell1 = tf.nn.rnn_cell.LSTMCell(num_units=120, cell_clip=None, initializer=initK)

        
#         # Unidirectional
#         R_out, State = tf.nn.dynamic_rnn(Cell1, H, initial_state=None,dtype=tf.float32,time_major=False)
#         H = R_out[:,-1,:]
#         # dim N*120

        # Bidirectional
        (R_f,R_b), State = tf.nn.bidirectional_dynamic_rnn(Cell1, Cell1, H, initial_state_fw=None,initial_state_bw=None, dtype=tf.float32,time_major=False)
        
        R_f = R_f[:,-1,:]
        R_b = R_b[:,-1,:]
        H = tf.concat([R_f,R_b],axis=1)
        # dim N*(120*2)

        
#         H = tf.layers.dense(H,units=4,activation=tf.nn.elu,kernel_initializer=initK)
        H = tf.layers.dense(H,units=4,activation=None,kernel_initializer=initK)

        # dim N*4
        
        Y_pred = tf.nn.softmax(H,name='Y_pred') 
        Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y_pred, labels=Y))
        Update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(Update_ops):
            Train_step = tf.train.AdamOptimizer(LR).minimize(Loss)
        init2 = tf.global_variables_initializer()
        
        
        
        ### training model ###
        batch_size = train_len
        lr=0.005

        loss_hist   = []
        loss_hist2  = []
        train_hist  = []
        test_hist2  = []


        with tf.Session(graph=gf) as se:
            se.run(init2)
            for epoch in range(100):
                if epoch >10:
                    lr *= 0.975
                id_list = get_batch_id(batch_size,train_len)
                for batch_id in id_list:
                    batch_x = x_train[batch_id]
                    batch_y = y_train[batch_id]    
                    _ , loss_i =  se.run([Train_step,Loss] , feed_dict={X: batch_x, Y: batch_y, T: True, LR:lr})

                loss_test = se.run(Loss,feed_dict={X: x_test ,Y:y_test , T:False })

                y_test_pred      = se.run(Y_pred, feed_dict={X: x_test, T: False})
                Y_correct_test   = tf.equal(tf.argmax(y_test_pred,1), tf.argmax(y_test,1))
                Y_correct_test   = tf.cast(Y_correct_test, tf.float32)
                acc_tf_test      = tf.reduce_mean(Y_correct_test)             
                acc_test2        = se.run(acc_tf_test)   

                Y_train_pred     = se.run(Y_pred, feed_dict={X: batch_x, Y: batch_y, T: False})
                Y_correct_train  = tf.equal(tf.argmax(Y_train_pred,1), tf.argmax(batch_y,1))
                Y_correct_train  = tf.cast(Y_correct_train, tf.float32)
                acc_tf_train     = tf.reduce_mean(Y_correct_train)             
                acc_train        = se.run(acc_tf_train)   

                loss_hist.append(loss_i)
                loss_hist2.append(loss_test)
                train_hist.append(acc_train)
                test_hist2.append(acc_test2)

                print('Epoch:', epoch, '| test: %.4f' % acc_test2, '| train: %.4f' % acc_train,
                      '| train Loss: %.4f' % loss_i, '| test Loss: %.4f' % loss_test)

        plt.figure()
        plt.plot(train_hist,color='red',label='train')
        plt.plot(test_hist2,color='green',label='test')

        plt.ylabel('Test_acc , Train_acc')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(loss_hist, color='red',label='train loss')
        plt.plot(loss_hist2,color='green',label='test  loss')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


""" ConvNet Nocrop """
def ConvNet_nocrop(data_list):

    [train_len, test_len ,x_train, y_train, x_test , y_test] = data_list
    
    #Define graph
    gf = tf.Graph()

    with gf.as_default():
        initK = tf.contrib.layers.xavier_initializer()
        #Placeholders
        X  = tf.placeholder(dtype=tf.float32,shape=(None,1000,22,1),name='X')
        Y  = tf.placeholder(dtype=tf.float32,shape=None,name='Y')
        T  = tf.placeholder(tf.bool)
        LR = tf.placeholder(tf.float32)

        H = tf.layers.conv2d(X,filters=40,kernel_size=(25,1),strides=(1, 1),padding='valid',activation=None ,kernel_initializer=initK)
        H = tf.layers.batch_normalization(H,momentum=0.1,training=T) 
        # dim N*976*22*40

        H = tf.layers.conv2d(H,filters=40,kernel_size=(1,22),strides=(1, 1),padding='valid',activation=None ,kernel_initializer=initK)
        H = tf.layers.batch_normalization(H,momentum=0.1,training=T) 
        # dim N*976*1*40

        H = tf.square(H)
        H = tf.layers.average_pooling2d(H , pool_size=(75,1) , strides=(15, 1), padding='valid')
        # dim N*61*1*40 !!!!
        
        H = tf.log(H)
        H = tf.layers.dropout(H,rate=0.5,training=T)

        H = tf.layers.conv2d(H,filters=4,kernel_size=(61, 1),padding='valid',activation=None ,kernel_initializer=initK)   
        # dim (N)*1*1*4

        Y_pred = tf.reshape(H,[-1,4],name='Y_pred')
        # dim (N)*4

        Y_pred = tf.nn.softmax(Y_pred,name='Y_pred')  # w w/o result simular
        Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y_pred, labels=Y))
        Update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(Update_ops):
            Train_step = tf.train.AdamOptimizer(LR).minimize(Loss)

        init2 = tf.global_variables_initializer()
        
        
        ### training model ###
        batch_size = train_len
        lr=0.005

        loss_hist   = []
        loss_hist2  = []
        train_hist  = []
        test_hist2  = []

        with tf.Session(graph=gf) as se:
            se.run(init2)
            for epoch in range(60):
                if epoch >10:
                    lr *= 0.975
                id_list = get_batch_id(batch_size,train_len)
                for batch_id in id_list:
                    batch_x = x_train[batch_id]
                    batch_y = y_train[batch_id]    
                    
                    _ , loss_i =  se.run([Train_step,Loss] , feed_dict={X: batch_x, Y: batch_y, T: True, LR:lr})

                loss_test = se.run(Loss,feed_dict={X: x_test ,Y:y_test , T:False })

                y_test_pred      = se.run(Y_pred, feed_dict={X: x_test, T: False})
                Y_correct_test   = tf.equal(tf.argmax(y_test_pred,1), tf.argmax(y_test,1))
                Y_correct_test   = tf.cast(Y_correct_test, tf.float32)
                acc_tf_test      = tf.reduce_mean(Y_correct_test)             
                acc_test2        = se.run(acc_tf_test)   


                Y_train_pred     = se.run(Y_pred, feed_dict={X: batch_x, Y: batch_y, T: False})
                Y_correct_train  = tf.equal(tf.argmax(Y_train_pred,1), tf.argmax(batch_y,1))
                Y_correct_train  = tf.cast(Y_correct_train, tf.float32)
                acc_tf_train     = tf.reduce_mean(Y_correct_train)             
                acc_train        = se.run(acc_tf_train)   

                loss_hist.append(loss_i)
                loss_hist2.append(loss_test)
                train_hist.append(acc_train)
                test_hist2.append(acc_test2)

                print('Epoch:', epoch, '| test: %.4f' % acc_test2, '| train: %.4f' % acc_train,
                      '| train Loss: %.4f' % loss_i, '| test Loss: %.4f' % loss_test)

        plt.figure()
        plt.plot(train_hist,color='red',label='train')
        plt.plot(test_hist2,color='green',label='test')

        plt.ylabel('Test_acc , Train_acc')
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(loss_hist, color='red',label='train loss')
        plt.plot(loss_hist2,color='green',label='test  loss')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()




""" ConvNet """

 
def ConvNet(data_list,epoch_all,batch_size,lr_decay):
    
    [train_len, test_len ,x_train, y_train, x_test , y_test , y_test_crop] = data_list
    
    if batch_size== None:
        batch_size = train_len
    if lr_decay==None:
        lr_decay = 0.975

    
    #Define graph
    gf = tf.Graph()

    with gf.as_default():
        initK = tf.contrib.layers.xavier_initializer()
        #Placeholders
        X  = tf.placeholder(dtype=tf.float32,shape=(None,1000,22,1),name='X')
        Y  = tf.placeholder(dtype=tf.float32,shape=None,name='Y')
        T  = tf.placeholder(tf.bool)
        LR = tf.placeholder(tf.float32)


        H = tf.layers.conv2d(X,filters=40,kernel_size=(25,1),strides=(1, 1),padding='valid',activation=None ,kernel_initializer=initK)
        H = tf.layers.batch_normalization(H,momentum=0.1,training=T) 
        # dim N*976*22*40

        H = tf.layers.conv2d(H,filters=40,kernel_size=(1,22),strides=(1, 1),padding='valid',activation=None ,kernel_initializer=initK)
        H = tf.layers.batch_normalization(H,momentum=0.1,training=T) 
        # dim N*976*1*40

        H = tf.square(H)
        H = tf.layers.average_pooling2d(H , pool_size=(75,1) , strides=(1, 1), padding='valid')
        # dim N*902*1*40

        #cropping in feature space 467 times samples
        Idx = tf.constant( np.arange(30) * 15 + np.arange(467)[:,np.newaxis] )
        H   = tf.gather(H,Idx,axis=1,name='Cropping')
        H   = tf.reshape(H,[-1,30,1,40])
        # dim (N*467)*30*1*40

        H = tf.log(H)
        H = tf.layers.dropout(H,rate=0.5,training=T)


        H = tf.layers.conv2d(H,filters=4,kernel_size=(30, 1),padding='valid',activation=None ,kernel_initializer=initK)   
        # dim (N*467)*1*1*4

        Y_pred = tf.reshape(H,[-1,4],name='Y_pred')
        # dim (N*467)*4

    #     Y_pred = tf.nn.softmax(Y_pred,name='Y_pred')

        Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y_pred, labels=Y))




        Update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(Update_ops):
            Train_step = tf.train.AdamOptimizer(LR).minimize(Loss)

        init2 = tf.global_variables_initializer()

        
    ### training model ###
    lr=0.005

    loss_hist   = []
    loss_hist2  = []
    train_hist  = []
    test_hist1  = []
    test_hist2  = []

    with tf.Session(graph=gf) as se:
        se.run(init2)
        for epoch in range(epoch_all):
            if epoch >10:
                lr *= lr_decay

            id_list = get_batch_id(batch_size,train_len)
            for batch_id in id_list:
                batch_x = x_train[batch_id]
                batch_y = y_train[batch_id]    
                batch_y_crop = np.repeat(batch_y,467,axis=0)    
                _ , loss_i =  se.run([Train_step,Loss] , feed_dict={X: batch_x, Y: batch_y_crop, T: True, LR:lr})

            y_test_pred  = se.run(Y_pred, feed_dict={X: x_test , T:False })
            y_test_pred  = se.run(tf.argmax(y_test_pred,1)).reshape(-1,467)    
            y_test_pred  = stats.mode(y_test_pred,axis=1)[0].reshape(test_len)  # vote
            y_correct    = np.equal(y_test_pred,y_test).astype(int)
            acc_test1    = y_correct.sum()/test_len
            loss_test = se.run(Loss,feed_dict={X: x_test ,Y:y_test_crop , T:False })

            y_test_pred      = se.run(Y_pred, feed_dict={X: x_test, T: False})
            Y_correct_test   = tf.equal(tf.argmax(y_test_pred,1), tf.argmax(y_test_crop,1))
            Y_correct_test   = tf.cast(Y_correct_test, tf.float32)
            acc_tf_test      = tf.reduce_mean(Y_correct_test)             
            acc_test2        = se.run(acc_tf_test)   

            Y_train_pred     = se.run(Y_pred, feed_dict={X: batch_x, Y: batch_y_crop, T: False})
            Y_correct_train  = tf.equal(tf.argmax(Y_train_pred,1), tf.argmax(batch_y_crop,1))
            Y_correct_train  = tf.cast(Y_correct_train, tf.float32)
            acc_tf_train     = tf.reduce_mean(Y_correct_train)             
            acc_train        = se.run(acc_tf_train)   

            loss_hist.append(loss_i)
            loss_hist2.append(loss_test)
            train_hist.append(acc_train)
            test_hist1.append(acc_test1)
            test_hist2.append(acc_test2)

            print('Epoch:', epoch, '| test1: %.4f' % acc_test1, '| test2: %.4f' % acc_test2, '| train: %.4f' % acc_train,
                  '| train Loss: %.4f' % loss_i, '| test Loss: %.4f' % loss_test)



    plt.figure()
    plt.plot(train_hist,color='red',label='train')
    plt.plot(test_hist1,color='green',label='test1')
    plt.plot(test_hist2,color='blue',label='test2')

    plt.ylabel('Test_acc , Train_acc')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(loss_hist, color='red',label='train loss')
    plt.plot(loss_hist2,color='green',label='test  loss')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
        
        
        
        

