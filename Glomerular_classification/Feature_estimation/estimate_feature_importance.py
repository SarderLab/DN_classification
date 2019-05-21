import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold

from data_loader import import_data,import_labels
from rnn_train import train_network
from rnn_predict_e import predict_holdout_e

def make_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory) # make directory if it does not exit already # make new directory
# Select cuda device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Location of features and labels
txt_loc='/hdd/BG_projects/JASN/RNN/Final_codes/KTRC-VUMC_Full_2.txt'
lbl_loc='/hdd/BG_projects/JASN/RNN/Final_codes/KTRC-VUMC_Full_labels.txt'
# Place to save models
model_path='/hdd/BG_projects/JASN/RNN/Final_codes/Feature_selection_models'
make_folder(model_path)
#Place to save feature prediction data
out_text=model_path+'/feature_predictions.txt'
f=open(out_text,'w')
f.close()

# Load features and labels
all_features = np.array(import_data(txt_loc))
labels=np.array(import_labels(lbl_loc))
# Number of input features to the network
num_input=len(all_features[0][0])
print(num_input)

# Training hyperparameters
learning_rate = 0.001
training_steps = 1000
batch_size = 256
save_interval=training_steps
# Length of glomerular sequences
glom_samplings=30
drop=0.5

# Number of glomerular sequence shuffles for prediction
predict_batch=1024
num_classes = np.max(np.array(labels))
print(num_classes)
print(len(all_features))


predictions=[]

train_index=range(0,len(all_features))
# Manually holdout some cases to ensure we aren't overfitting
test_index=[0,16,30,37]

for t in test_index[::-1]:
    train_index.pop(t)


# Train the network
saved_model_path=train_network(train_index,test_index,txt_loc,lbl_loc,learning_rate,training_steps,batch_size,
    save_interval,num_classes,model_path,glom_samplings,drop)
full_index=range(0,len(all_features))

# Get the predictions on all data without dropping any features (base predictions)
tf.reset_default_graph()
base_pred=predict_holdout_e(full_index,txt_loc,lbl_loc,[],saved_model_path,predict_batch)
# Write them to the output file
for idx,case in enumerate(base_pred):
    m_pred=np.mean(case)
    with open(out_text,'a+') as f:
        f.write(str(m_pred)+',')
with open(out_text,'a+') as f:
    f.write('\n')
for idx,case in enumerate(labels):
    with open(out_text,'a+') as f:
        f.write(str(case)+',')
with open(out_text,'a+') as f:
    f.write('\n')

# Iteratively drop each feature from the input features, predict on the entire dataset, and write the output to text file
p_stats=[]
for f_ in range(0,num_input):
    tf.reset_default_graph()
    print(f_)
    p=predict_holdout_e(full_index,txt_loc,lbl_loc,f_,saved_model_path,predict_batch)
    predictions.append(p)
    l=labels[test_index]

    for idx,case in enumerate(p):
        m_pred=np.mean(case)
        p_stats.append(m_pred)

        with open(out_text,'a+') as f:
            f.write(str(m_pred)+',')
    with open(out_text,'a+') as f:
        f.write('\n')
