# DN_classification

This repository contains the source codes for the publication entitled "Computational segmentation and classification of diabetic glomerulosclerosis", which was submitted after revision to JASN on XX. 

# DeepLab V2 network
In our work we use the DeepLab V2 Tensorflow implementation available here: https://github.com/zhengyang-wang/Deeplab-v2--ResNet-101--Tensorflow

# Data / Pre-trained models
Whole slide DN biopsy images, and trained glomerulus and nucleus segmentation models are here: https://buffalo.box.com/s/e40wzg2flb3p0r73zyhelhqvhle46vvr

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
You will need to download the modified model.py and main_n.py if you want to use the weighting scheme derived in the paper  

GCA and feature analysis:  
MATLAB (https://www.mathworks.com/products/matlab.html)  

RNN classifier:  
NumPy (https://www.numpy.org/)  
Tensorflow (https://www.tensorflow.org/)  
scikit-learn (https://pypi.org/project/sklearn/)  

# Contents
Glomerular_nuclear_detection:  
This directory contains a modified "main_n.py" and "model.py", which are needed to reproduce the network output weighting scheme presented in the paper. To use, download these files and place them in the respective DeepLab folder, overwriting the base copies of main.py and model.py. Then, to apply weighting, change the value of "prior" listed under the prediction parameters in "main.py".  
This directory also contains a file named xml_to_nuclei.py, which takes as input a folder of WSIs and XML annotations, and provides as output a formatted directory containing extracted glomeruli images with boundary and nuclei segmented. 

GCA_and_features:  
Human - Contains MATLAB codes for extracting glomerular features. Takes as input a structured directory output from xml_to_nuclei.py and goes through each patient folder and extracts the glomerular components and glomerular features. Writes all features to a formatted text file which can be used for RNN classification. To use, first acquire glomeruli using the xml_to_nuclei.py script, then run DN_classification_master.m and select the directory where glomerular images are located. 

Mouse - Contains MATLAB codes for extracting glomerular features, as mentioned directory above, only for mouse data instead. The main script is Mouse_classification_master.m

Glomerular_classification:  
Classification - Contains the algorithms necessary to classify glomerular features using RNN-based strategy presented in the paper. Specifically, the scripts will perform 10-folds of training and testing and save the predicted results in each holdout set to a desired text file. Takes as input a feature description text file and a set of labels. Yields cross-validated training models and predictions for each holdout set.  
Feature estimation - Contains the algorithms necessary to train a single model and perform sequential feature dropout at prediction time. Takes as input a feature description text file, a set of labels, and yields a formatted text file that contain the prediction data from each dropout iteration. 

Feature_texts:  
This directory contains pre-derived feature texts and labels which correspond to the experiments described in our manuscript. 

# Usage
Glomerular boundary detection and glomerular nucleus segmentation:  

Install DeepLab V2 and download pretrained models for low and high resolution glomerular segmentation, and high resolution nuclear segmentation. If you want to add a weighting amount to the nuclear segmentation outputs, modify the "prior" input variable found in the main_n.py script. Create a single folder in your workspace containing: 1) whole slide images in .svs format, 2) .xml annotation files for each .svs file. The syntax to call the glomerular and nuclear segmentation is:
    
    python xml_to_nuclei.py --wsi path/to/wsi/files/ --output path/to/desired/output/folder/
    
There are various parameters within the script which can be modified for alternative functionality, and they are all found at the beginning. 
    
    
Glomerular component analysis and feature extraction:  

Make sure MATLAB is properly installed and configured. For human analysis, run the script "DN_classification_master.m", which will prompt you to select a directory of patient data. The scripts are expecting the formatted directory structure created by the script xml_to_nuclei.py. For mouse analysis, run the script Mouse_classification_master.m. All modifiable parameters are found at the beginning of each script file. A string variable on line 19 of DN_classification_master.m determines where the output feature data should be stored  
    
Glomerular classification:

This algorithm takes as input a text file describing features for each parent, formatted in the style which is output from the glomerular component analysis and feature extraction algorithms, and a set of labels for each patient data, also in a text file. Sample feature texts and labels which were derived in the work are available in the "Feature_texts" directory. There are several important variables defined at the beginning of the script such as the feature location, label location, output model path, desired GPU device, etc. Once the proper information is supplied, running the algorithm,
    
    python KFoldRNN.py
    
will split the dataset into 10 folds for training and testing, and proceed to train 10 cross-validated models. At the end of training on each fold, predictions will be acquired for the holdout set of the current fold. All predictions and associated ground truth labels for all 10 folds will be written to a text file upon completion of cross validation (location specified by user).
