import sys
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from data_loader import import_data,random_batcher,import_labels

def predict_holdout(test_index,txt_loc,lbl_loc,model_path,num_classes,batch_size):

    # Import feature and labels
    all_features = np.array(import_data(txt_loc))
    labels=np.array(import_labels(lbl_loc))

    # Network parameters
    num_hidden = 50
    num_hidden_2 = 25


    # Get data for standardization
    mean_record=[]
    std_record=[]
    for case in all_features:
        mean_record.append(np.mean(np.asarray(case),0))
        std_record.append(np.std(np.asarray(case),0))
    feature_mean=np.nanmean(mean_record,0)
    feature_std=np.nanstd(std_record,0)

    # Pull the holdout features
    holdout_features=all_features[test_index]
    holdout_labels=labels[test_index]

    # Number of input features
    num_input=len(holdout_features[0][0])

    # Unroll each set of features so that we have a vector of all glomeruli for each case
    holdout_l=holdout_labels
    holdout_f=holdout_features
    holdout_f_vector=[]
    for case in range(0,len(holdout_f)):
        case_features=np.zeros((len(holdout_f[case]),len(holdout_f[case][0])))
        for glom in range(0,len(holdout_f[case])):
            case_features[glom,:]=holdout_f[case][glom]
        holdout_f_vector.append(case_features)

    # Placeholders for graph input
    X = tf.placeholder(tf.float32, [None, None, num_input])
    Y = tf.placeholder(tf.float32, [None, 1])

    dropout2 = tf.placeholder(tf.float32,shape=(),name='dropout2')

    # Function to set up LSTM
    def lstm(x_,unit_num,seq_flag):
        lstm_cell =tf.keras.layers.LSTMCell(unit_num, unit_forget_bias=True)
        lstm_layer= tf.keras.layers.RNN(lstm_cell,return_sequences=seq_flag, dtype=tf.float32)
        output=lstm_layer(x_)
        return output

    # For performance analysis
    def MSE(x,y):
        return np.sum((x-y)**2)/batch_size

    # Dense layer to choose input importance
    D_in=tf.layers.dense(X,num_input,activation=tf.nn.leaky_relu)

    # First LSTM unit
    cell_1 = lstm(x_=D_in,unit_num=num_hidden,seq_flag=True)
    # Inter-LSTM dropout
    cell_1=tf.nn.dropout(cell_1,dropout2)
    # Second LSTM unit
    cell_2 = lstm(x_=cell_1,unit_num=num_hidden_2,seq_flag=False)
    # The predicted value
    prediction=tf.layers.dense(cell_2,1,activation=tf.nn.leaky_relu)
    # A clipped version of the predicted value
    clipped_prediction=tf.clip_by_value(prediction,1,num_classes)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(abs( prediction - Y) )

    # Performance metric
    MSE_ = tf.reduce_mean(MSE(prediction,Y))

    # Initialize saver object to restore the model
    saver=tf.train.Saver()

    with tf.Session() as sess:
        # Restore the model
        saver.restore(sess,model_path)
        # Place to keep predicted values
        predictions=np.zeros((len(holdout_f_vector),batch_size))

        # For all cases
        for step in range(0,len(holdout_f_vector)):
            # Get the holdout cases
            holdout_idx=test_index[step]
            holdout_batch=np.zeros((batch_size,len(holdout_f_vector[step]),num_input))
            holdout_label=np.zeros((batch_size,1))

            # Randomly permute the sequence of gloms to create a batch of testing sequences
            for step_i in range(0,batch_size):
                holdout_batch[step_i,:,:]=holdout_f_vector[step]
                np.random.shuffle(holdout_batch[step_i,:,:])
                holdout_label[step_i,0]=labels[holdout_idx]
            # Standardize
            holdout_batch-=feature_mean
            holdout_batch/=feature_std
            holdout_batch=np.nan_to_num(holdout_batch)
            val_loss,val_acc,c_p = sess.run([loss_op,MSE_,clipped_prediction], feed_dict={X: holdout_batch, Y: holdout_label, dropout1: 1, dropout2: 1})

            predictions[step,:]=np.squeeze(c_p)
    return predictions
