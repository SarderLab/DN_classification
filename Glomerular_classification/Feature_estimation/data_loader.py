import numpy as np
import random

def import_data(txt_file):
    f = open(txt_file, 'r')
    lines = f.readlines()
    f.close()
    lines = np.array(lines)
    feats=[]
    features=[]
    for line in lines:

        if "---" in line:
            feats.append(features)
            features=[]
            continue


        featurevals=line.split(',')
        f_=[]
        for f_val in featurevals[:-1]:
            f_.append(float(f_val))
        features.append(f_)
    return feats

def import_labels(lbl_file):
    f = open(lbl_file, 'r')
    lines = f.readlines()
    f.close()
    labels=[]
    labels_txt=lines[0].split(',')
    labels=np.zeros(len(labels_txt))
    for l in range(0,len(labels_txt)):
        labels[l]=labels_txt[l]
    return labels

def random_batcher(feats,time_step,batch_size,labels):

    num_cases=len(feats)
    num_features=len(feats[0][1])
    batched_data=np.zeros( ( batch_size,time_step,len(feats[0][1]) ) )
    batched_labels=np.zeros((batch_size,1))
    for i in range(0,batch_size):
        random_idx=random.randrange(num_cases)
        random_case=feats[random_idx]
        random_label=labels[random_idx]

        batched_labels[i]=random_label
        for j in range(0,time_step):
            random_glom = random.choice(random_case)

            for v in range(0,num_features):
                batched_data[i,j,v]=random_glom[v]

    return batched_data,batched_labels
