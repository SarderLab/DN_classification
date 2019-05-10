function processDeepLabSegmentations(nuc_dir)

parfor q=1:length(nuc_dir)
    mask=imread(fullfile(nuc_dir(q).folder,nuc_dir(q).name));
    % Outputs from deeplab are uint8 grayscale images, threshold for a
    % logical image before passing to the splitting function
    mask=split_nuclei_functional(mask(:,:,1)>0);
    % Overwrite nuclear segmentation with the processed version
    imwrite(mask,fullfile(nuc_dir(q).folder,nuc_dir(q).name));
end