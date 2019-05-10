close all
clear all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Developed by Brandon Ginley while researching as a PhD candidate in the 
% lab of Pinaki Sarder at the SUNY Jacobs School of Medicine and Biomedical
% Sciences.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define folder containing image data for each case
case_dir=uigetdir();
% List the cases
biopsy_cases=dir(case_dir);
biopsy_cases(1:2)=[];

dirFlags=[biopsy_cases.isdir];
biopsy_cases=biopsy_cases(dirFlags);

% Where to store the features
txt=fopen('VUMC_full.txt','a');

% For all cases
for b_c=1:length(biopsy_cases)
    % Get case identifier and associated image folders
    case_ID=biopsy_cases(b_c).name;
    display(['Working on case ' case_ID])
    
    image_dir=dir(fullfile(case_dir,case_ID,'/Images/*.png'));
    boundary_dir=dir(fullfile(case_dir,case_ID,'/Boundary_segmentations/*.png'));
    nuc_dir=dir(fullfile(case_dir,case_ID,'/Nuclear_segmentations/prediction/*.png'));

    % Perform distance transform estimation of clumped nuclei
    % and marker-controlled watershed splitting of multiple nuclei
    processDeepLabSegmentations(nuc_dir)

    % Create directory for output glomerular compartment segmentations
    segment_out_dir=[case_dir,'/',case_ID,'/CompartmentSegmentations'];
    if ~exist(segment_out_dir)
       mkdir(segment_out_dir) 
    end
    
    % Generate glomerular compartment segmentations
    glomerularCompartmentSegmentation(image_dir,boundary_dir,nuc_dir,segment_out_dir)

    % Perform feature extraction on glomerular compartment segmentation
    % maps
    features=glomerularFeatureExtraction(segment_out_dir,image_dir);

    % Create directory for extracted features and save them
    feature_out_dir=[case_dir,'/',case_ID,'/Features'];
    if ~exist(feature_out_dir)
       mkdir(feature_out_dir) 
    end
    
    % Save the features as a matlab matrix file
    save([feature_out_dir,'/',case_ID],'features')

    % Write the features to text file
    [d1,d2]=size(features);
    for i=1:d1
         for j=1:d2
            fprintf(txt,[num2str(features(i,j)),',']);
         end
         fprintf(txt,'\n');
        if i==d1
         fprintf(txt,'---\n');
        end
    end

end
fclose('all')