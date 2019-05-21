import numpy as np
import os
import cv2
import openslide
import lxml.etree as ET
import sys
import argparse
import warnings

from glob import glob
from skimage.io import imsave
from subprocess import call

cwd = os.getcwd() + '/'
# Regions with fewer pixels than size_thresh are not included
size_thresh = 3000
# Amount to pad each image dimension around glomerular boundary in extracted images
pad_width=100
#Image extension
imBoxExt='.png'
#Which device to use
gpu_id='0'

#Directory of tensorflow model and model checkpoint number
model_dir='Deeplab-v2--ResNet-101--Tensorflow-master/model'
nuc_ckpt=300000

# Where the deeplab folder is located
deeplab_dir=cwd+'/Deeplab-v2--ResNet-101--Tensorflow-master'


WSIs = []
XMLs = []
# Class ID
annot_ID=1



def get_annotation_bounds(xml_path, annotationID=1):
    # parse xml and get root
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Find listed regions
    Regions = root.findall("./Annotation[@Id='" + str(annotationID) + "']/Regions/Region")

    masks = []
    extremas=[]
    # Create padded mask and identify boundary extremas for all regions
    for Region in Regions:
        Vertices = Region.findall("./Vertices/Vertex")
        x = []
        y = []

        for Vertex in Vertices:
            x.append(int(np.float32(Vertex.attrib['X'])))
            y.append(int(np.float32(Vertex.attrib['Y'])))

        x1=min(x)
        x2=max(x)
        y1=min(y)
        y2=max(y)
        points = np.stack([np.asarray(x), np.asarray(y)], axis=1)

        points[:,1] = np.int32(np.round(points[:,1] - y1 ))
        points[:,0] = np.int32(np.round(points[:,0] - x1 ))

        mask = np.zeros([(y2-y1),x2-x1], dtype=np.int8)

        # Fill mask boundary regions
        cv2.fillPoly(mask, [points], 1)
        mask=np.pad( mask,(pad_width,pad_width),'constant',constant_values=(0,0) )

        masks.append(mask)
        extremas.append([x1,x2,y1,y2])
    return masks,extremas

def make_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory) # make directory if it does not exit already # make new directory

def restart_line(): # for printing chopped image labels in command line
    sys.stdout.write('\r')
    sys.stdout.flush()

def getWsi(path): #imports a WSI
    import openslide
    slide = openslide.OpenSlide(path)
    return slide

def file_len(fname): # get txt file length (number of lines)
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

# Get the information input from the user
parser = argparse.ArgumentParser()
parser.add_argument('--wsi', dest='wsi', default=' ',type=str, help='Specifies the whole slide folder path.')
parser.add_argument('--output', dest='output_directory_name', default=' ',type=str, help='Directory to save output results.')
args = parser.parse_args()

# Make sure user gives information
if args.output_directory_name==' ':
    print('No output directory provided, using default directory at: '+'\n')
    output_directory_name='Output/'
    outDir=cwd+output_directory_name
else:
    outDir=cwd+'/'+args.output_directory_name+'/'

# check main directory exists, if not, make it
make_folder(outDir)

# Make sure user gives information
if args.wsi == ' ':
    print('\nPlease specify the whole slide folder path.\n\nUse flag:')
    print('--wsi <path>\n')
    sys.exit()
# Get list of all whole slide images
WSIs_ = glob(args.wsi+'/*.svs')
for WSI in WSIs_:
    xml_ = glob(WSI.split('.')[0] + '.xml')
    if xml_ != []:
        print('including: ' + WSI)
        XMLs.append(xml_[0])
        WSIs.append(WSI)

# go though all WSI
for idx, XML in enumerate(XMLs):
    # Generate mask region and boundary maxima from XML annotation
    masks,extremas = get_annotation_bounds(XML,annot_ID)

    basename = os.path.basename(XML)
    basename = os.path.splitext(basename)[0]

    # Create output folders
    make_folder(outDir + basename+'/Boundary_segmentations')
    make_folder(outDir + basename+ '/Images')
    # Open wholeslide image data
    print('opening: ' + WSIs[idx])
    pas_img = openslide.OpenSlide(WSIs[idx])

    # For all discovered regions in XML
    for idxx, ex in enumerate(extremas):
        mask = masks[idxx]
        size=np.sum(mask)
        if size >= size_thresh:
            # Pull image from WSI
            c_1=ex[0]-pad_width
            c_2=ex[2]-pad_width
            l_1=(ex[1]+pad_width)-c_1
            l_2=(ex[3]+pad_width)-c_2
            PAS = pas_img.read_region((c_1,c_2), 0, (l_1,l_2))
            PAS = np.array(PAS)[:,:,0:3]

            print(basename + '_' + str(idxx))
            # Save image and mask
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imsave(outDir + basename+ '/' + 'Images/' + basename + '_' + str(idxx) + '.png',PAS)
                imsave(outDir + basename +'/' + 'Boundary_segmentations/' + basename + '_' + str(idxx) + '_mask.png', mask*255,check_contrast=False)
    pas_img.close()

# Begin nuclear prediction
# Get list of all cases generated by the xml extraction
input_folder_list=glob(outDir+'/*'+os.path.sep)

for case_folder in input_folder_list:
    # Get case images
    case_folder=case_folder.split('/')[-2]
    images_for_prediction=glob(outDir+'/'+case_folder+'/Images/*.png')
    # Make output folder
    make_folder(outDir+'/'+case_folder+'/Nuclear_segmentations/prediction')
    # Where to save test image names for DeepLab prediction
    txt_loc=deeplab_dir+'/dataset/test.txt'
    f=open(txt_loc,'w')
    f=open(txt_loc,'a')
    # Write image names to text
    for image in images_for_prediction:
        im_splits=image.split('/')
        file_ID=im_splits[-1]
        folder='/'+im_splits[-2]+'/'

        f.write(folder+file_ID+'\n')
    f.close()
    # Call deeplab for prediction
    call(['python3', deeplab_dir+'/main_n.py',
        '--option', 'predict',
        '--test_data_list', txt_loc,
        '--out_dir', outDir+'/'+case_folder+'/Nuclear_segmentations/',
        '--test_step', str(nuc_ckpt),
        '--test_num_steps', str(len(images_for_prediction)),
        '--modeldir', model_dir,
        '--data_dir', outDir+case_folder,
        '--gpu', gpu_id])
