function glomerularCompartmentSegmentation(image_dir,boundary_dir,nuc_dir,out_dir)


parfor g=1:length(image_dir)

    % Read image, glomerular mask, and nuclear mask
    I=imread(fullfile(image_dir(g).folder,image_dir(g).name));

    % Identify unique image name to properly read associated masks from
    % other folders
    % Threshold at zero because outputs directory from DeepLab are uint8
    % intensity images rather than logicals
    uID=strsplit(image_dir(g).name,'.png');
    boundary=imread(fullfile(boundary_dir(g).folder,[uID{1,1},'_mask.png']))>0;
    nucSeg=imread(fullfile(nuc_dir(g).folder,[uID{1,1},'_mask.png']))>0;

    % Perform LAB transformation
    lab_tf=makecform('srgb2lab');
    
    LAB=im2double(applycform(I,lab_tf));

    % Threshold white space using Otsu's method
    lightness=(LAB(:,:,1));
    WhiteSpaces=(imbinarize(lightness,graythresh(lightness)));

   
     [a,b,~]=colour_deconvolution(I,'H PAS');

    mesInt=1-im2double(b);

    % Find mesangial segmentation on PAS-stain deconvolved channel using Otsu's
    % method
    mes=imbinarize(mesInt,graythresh(mesInt));

    % Eliminate non-glomerular nuclei
    nucSeg(~boundary)=0;
    
    % Crop to glomerular boundary
    WhiteSpaces(~boundary)=0;
    % Crop PAS-stain components to glomerular boundary
    mes(~boundary)=0;

    % Remove nuclear regions from mesangial segmentation
%     mes(nucSeg)=0;
    % Remove nuclear regions from luminal segmentation
    WhiteSpaces(nucSeg)=0;

    % Make label image for training and testing Naive Bayes glomerular 
    % segmenter 
    onelabel=zeros(size(mes));
    onelabel(WhiteSpaces)=2;
    onelabel(mes)=1;
    
    % Extract regions that need to be labeled 
    testI=I;
    testI=double(reshape(testI,size(testI,1)*size(testI,2),[]));

    % Re-format training data into vector
    trainI=double(reshape(I,size(I,1)*size(I,2),[]));
    labelVector=reshape(onelabel,size(onelabel,1)*size(onelabel,2),1);
    

    testI(labelVector==0)=NaN;
    labelVector(labelVector==0)=NaN;
    try
    % Fit naive Bayesian classifier
    Mdl=fitcnb(trainI,labelVector);
        % Predict labels for testing pixels using trained naive Bayesian model
    otherlabel=predict(Mdl,testI);

    % Reshape predictions back into an image
    otherVis=(reshape(otherlabel,size(I,1),size(I,2)));

    % Finish segmentation map by placing final predictions over the old map
    onelabel(otherVis==1)=1;
    onelabel(otherVis==2)=2;
    onelabel(~boundary)=0;
    onelabel(nucSeg)=0;

    % Visualize / store segmentations
    mes=onelabel==1;

    WhiteSpaces=onelabel==2;
    mes(nucSeg)=0;
    WhiteSpaces(nucSeg)=0;
    final_mask=cat(3,mes,WhiteSpaces,nucSeg);
    final_mask(~repmat(boundary,[1,1,3]))=0;
    imwrite(double(final_mask),[out_dir,'/',image_dir(g).name])

    catch
        display(['Skipping compartment segmentation for image ' num2str(g)])
 
    end

end