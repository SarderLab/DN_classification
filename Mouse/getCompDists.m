function [compDists]=getCompDists(mask,gOutline,gCenter)
L=logical(mask);

%If compartment doesn't exist (e.g. no nuclei in a sclerotic glomerulus),
%skip it
if sum(sum(mask(:)))==0
    GCeCoDistance=0;
    GBCDistance=0;
    GCoCoDistance=0;
    compDists=['CenterDist',GCeCoDistance','MeanBoundaryDist',mean(GBCDistance,2), ...
    'MaxBoundaryDist',max(GBCDistance,[],2),'MinBoundaryDist', ...
    min(GBCDistance,[],2),'MeanNNDistance',mean(GCoCoDistance,2),'MaxNNDistance', ...
    max(GCoCoDistance,[],2),'MinNNDistance',min(GCoCoDistance,[],2)];
else


%Indices of glomerular boundary
[rPerim,cPerim]=find(gOutline);
%Centroids of compartment
s=regionprops(L,'Centroid');
compCenters=struct2table(s);
compCenters=[compCenters.Centroid];
%Pairwise distance between glomerular center and compartment centers
GCeCoDistance=pdist2(gCenter,compCenters);
%Pairwise distance between glomerular boundary and compartment centers
GBCDistance=pdist2(compCenters,[rPerim,cPerim]);
%Pairwise distance between compartment centers and themselves
GCoCoDistance=pdist2(compCenters,compCenters);

%Sort the data so we can extract the SECOND smallest distance (the actual 
%smallest distance is always 0 since we are taking the distance between a list of objects and itself)
GCoCoDistance=sort(GCoCoDistance);
if length(GCoCoDistance)==1
    GCoCoOut=GCeCoDistance;
else
   GCoCoOut= GCoCoDistance(2,:)';
end

compDists=[GCeCoDistance',mean(GBCDistance,2),max(GBCDistance,[],2), ...
    min(GBCDistance,[],2),mean(GCoCoDistance,2),max(GCoCoDistance,[],2), ...
    GCoCoOut];
end