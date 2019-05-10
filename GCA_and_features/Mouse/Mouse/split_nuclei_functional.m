function final_nuclei=split_nuclei_functional(nucleus_mask)

%Parameters
% For removing small objects which are not full nuclear objects
min_size=10; 

% Percentage of maximum distance transform values to be used as H-maxima
% when computing the extended-maxima transform
peak_est_thr=0.05;

% Amount to smooth distance transform peaks to better estimate clumps
gauss_sig=1;

% Remove very small objects
nuclear_individuals=bwareaopen(nucleus_mask,min_size);

% Disconnect border objects for compatibility with subsequent algorithms
nuclear_individuals(1,:)=0;
nuclear_individuals(end,:)=0;
nuclear_individuals(:,1)=0;
nuclear_individuals(:,end)=0;

% Calculate distance transform of nuclei to estimate peaks of clumped
% objects
ndt=bwdist(~nuclear_individuals);

% Perform light peak smoothing
ndtf=imgaussfilt(ndt,gauss_sig);

% Identify peaks of distance transform using a percentage of the maximum
% value of the gaussian-smoothed distance transform as the H-maxima
nuc_peaks=imextendedmax(ndtf,peak_est_thr*max(ndtf(:)));

% Identify the binary regions which contain multiple peaks, corresponding
% to multiple nuclei
multiple_nucs=bwpropfilt((nuclear_individuals-nuc_peaks)>0,'EulerNumber',[-100,-1]);
multiple_nucs=imfill(multiple_nucs,'holes');

% Remove and save single nuclear regions for later
single_nucs=nuclear_individuals-imfill(multiple_nucs,'holes');

% Identify skeleton by influence zones for marker-based watershed
mn_dt=bwdist(multiple_nucs);
mn_peaks=nuc_peaks&multiple_nucs;
mn_w=watershed(mn_dt);
ridgelines=mn_w==0;

% Perform marker guided watershed using estimated peaks and ridgelines as
% markers
guided_watershed=imimposemin(mn_dt,ridgelines|mn_peaks);
L=watershed(guided_watershed);
segmented_nuclei=multiple_nucs-(L==0)>0;
segmented_nuclei=bwareaopen(segmented_nuclei,min_size);

%Lightly round off the sharp corners produced by watershed splitting
segmented_nuclei=imopen(segmented_nuclei,strel('disk',1));

%Combine split nuclei and single nuclei
final_nuclei=segmented_nuclei|single_nucs;
