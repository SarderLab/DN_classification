# DN_classification

This repository contains the source codes for the publication entitled "Computational segmentation and classification of diabetic glomerulosclerosis", which was submitted after revision to JASN on XX. 

# DeepLab V2 ResNet Models
All glomerulus and nucleus segmentation models are available at https://buffalo.box.com/s/e40wzg2flb3p0r73zyhelhqvhle46vvr

# Requirements
Glomerular detection:  
XML annotation files for Aperio ImageScope whole slide images, either acquired using our HAIL pipeline for WSI segmentation (https://github.com/SarderLab/H-AI-L), or manual annotation  
OpenSlide (https://openslide.org/)  
DeepLab V2 for Tensorflow (https://github.com/zhengyang-wang/Deeplab-v2--ResNet-101--Tensorflow)
lxml (https://lxml.de/)  
OpenCV Python (3.4.0.12) (http://opencv-python-tutroals.readthedocs.io/en/latest/index.html)  
skimage (0.15.0) (http://scikit-image.org/docs/dev/api/skimage.html)  

Nuclear detection:
DeepLab V2 for Tensorflow (and also you will need to download the modified model.py and main.py if you want to use the weighting scheme derived in the paper)

GCA and feature analysis:  
MATLAB (https://www.mathworks.com/products/matlab.html)  

RNN classifier:  
NumPy (https://www.numpy.org/)  
Tensorflow (https://www.tensorflow.org/)  
scikit-learn (https://pypi.org/project/sklearn/)  
