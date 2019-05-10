function features=glomerularFeatureExtraction(segmentation_dir,image_dir)

segmented_gloms=dir([segmentation_dir,'/*.png']);

Total=length(segmented_gloms);
features=zeros(Total,218);
min_object_size=25;

parfor q=1:Total
    features(q,:)=feature_extraction_inner(q,segmented_gloms,image_dir,min_object_size);

end