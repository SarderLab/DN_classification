import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from data_loader import import_data,import_labels
from rnn_train import train_network
from rnn_predict import predict_holdout

def make_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory) # make directory if it does not exit already # make new directory
# choose GPU or CPU device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Define location of features and labels
txt_loc='/hdd/BG_projects/JASN/RNN/Final_codes/KTRC-VUMC_Full_2.txt'
lbl_loc='/hdd/BG_projects/JASN/RNN/Final_codes/KTRC-VUMC_Full_labels.txt'

# Define where to store saved models
model_path='/hdd/BG_projects/JASN/RNN/Final_codes/KTRC-VUMC_full'
make_folder(model_path)
# Define name of text file to write predictions to
out_text=model_path+'/VUMC_full_predictions.txt'
f=open(out_text,'w')
f.close()

# Import the features
all_features = np.array(import_data(txt_loc))
print(len(all_features))
print(len(all_features[0][0]))
# Import the labels
labels=np.array(import_labels(lbl_loc))

predictions=[]

p_stats=[]
fold=0
# Dropout between LSTM units
drop=0.5
# Initial learning rate
learning_rate = 0.001
# Number of steps to train
training_steps = 1000
# Size of each training batch
batch_size = 256
# Length of glomerular input sequence for training
glom_samplings=30
# Number of times to shuffle full-length glomerular sequences for prediction
predict_shuffles=1024
# How often to save model parameters, set to save only once
save_interval=training_steps
#Determine number of classes from labels
num_classes = np.max(np.array(labels))
print(num_classes)
kf=KFold(10,shuffle=True)

for train_index, test_index in kf.split(all_features):
    print('---------------------------------------------------')
    print('Current fold: ' +str(fold)+' ',("TEST:", test_index))
    print('---------------------------------------------------')
    tf.reset_default_graph()
    #Train the network and return the path of the saved model parameters
    saved_model_path=train_network(train_index,test_index,txt_loc,lbl_loc,learning_rate,training_steps,batch_size,
        save_interval,num_classes,model_path,glom_samplings,drop)
    tf.reset_default_graph()
    #Predict with the network on the holdout set and full-length glomerular sequences
    p=predict_holdout(test_index,txt_loc,lbl_loc,saved_model_path,num_classes,predict_shuffles)
    predictions.append(p)
    l=labels[test_index]

    # Save the holdout predictions for performance analysis
    for idx,case in enumerate(p):
        p_stats.append([np.mean(case),l[idx]])
    fold+=1
# Write predicitons to text file
with open(out_text,'a') as f:
    for l in range(0,len(p_stats)):
        f.write(str(p_stats[l][0])+','+str(p_stats[l][1])+'\n')
