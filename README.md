# DN_classification

This repository contains the source codes for the publication entitled "Computational segmentation and classification of diabetic glomerulosclerosis", which was submitted after revision to JASN on XX. 

# DeepLab V2 network
In our work we use the DeepLab V2 Tensorflow implementation available here: https://github.com/zhengyang-wang/Deeplab-v2--ResNet-101--Tensorflow

# Data / Pre-trained models
Whole slide DN biopsy images made available with this work can be found here: https://buffalo.box.com/s/e40wzg2flb3p0r73zyhelhqvhle46vvr

All trained glomerulus and nucleus segmentation models are available at https://buffalo.box.com/s/e40wzg2flb3p0r73zyhelhqvhle46vvr

# Requirements
Glomerular detection:  
XML annotation files for Aperio ImageScope whole slide images, either acquired using our HAIL pipeline for WSI segmentation (https://github.com/SarderLab/H-AI-L), or manual annotation  
OpenSlide (https://openslide.org/)  
DeepLab V2 for Tensorflow (https://github.com/zhengyang-wang/Deeplab-v2--ResNet-101--Tensorflow)  
lxml (https://lxml.de/)  
OpenCV Python (3.4.0.12) (http://opencv-python-tutroals.readthedocs.io/en/latest/index.html)  
skimage (0.15.0) (http://scikit-image.org/docs/dev/api/skimage.html)  
PIL (5.3.0) (https://pillow.readthedocs.io/en/latest/)  

Nuclear detection:  
If using the provided script xml_to_nuclei.py, you will need all of the requirements listed under Glomerular detection above.  
You will need to download the modified model.py and main.py if you want to use the weighting scheme derived in the paper  

GCA and feature analysis:  
MATLAB (https://www.mathworks.com/products/matlab.html)  

RNN classifier:  
NumPy (https://www.numpy.org/)  
Tensorflow (https://www.tensorflow.org/)  
scikit-learn (https://pypi.org/project/sklearn/)  

# Contents
Glomerular_nuclear_detection:  
This directory contains a modified "main.py" and "model.py", which are needed to reproduce the network output weighting scheme presented in the paper. To use, download these files and place them in the respective DeepLab folder, overwriting the base copies of main.py and model.py. Then, to apply weighting, change the value of "prior" listed under the prediction parameters in "main.py".  
This directory also contains a file named xml_to_nuclei.py, which takes as input a folder of WSIs and XML annotations, and provides as output a formatted directory containing extracted glomeruli images with boundary and nuclei segmented. 

GCA_and_features:  
Human - Contains MATLAB codes for extracting glomerular features. Takes as input a structured directory output from xml_to_nuclei.py and goes through each patient folder and extracts the glomerular components and glomerular features. Writes all features to a formatted text file which can be used for RNN classification. To use, first acquire glomeruli using the xml_to_nuclei.py script, then run DN_classification_master.m and select the directory where glomerular images are located. 

Mouse - Contains MATLAB codes for extracting glomerular features, as mentioned directory above, only for mouse data instead. The main script is Mouse_classification_master.m

Glomerular_classification:  
Classification - Contains the algorithms necessary to classify glomerular features using RNN-based strategy presented in the paper. Specifically, the scripts will perform 10-folds of training and testing data and save the predicted results in each holdout set to a desired text file. Takes as input a feature description text file and a set of labels. Yields cross-validated training models and predictions for each holdout set.  
Feature estimation - Contains the algorithms necessary to train a single model and perform sequential feature dropout at prediction time. Takes as input a feature description text file, a set of labels, and yields a formatted text file that contain the prediction data from each dropout. 

Feature_texts:  
This directory contains pre-derived feature texts and labels which correspond to the experiments described in our manuscript. 


