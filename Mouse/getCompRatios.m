function [ratios,s,compNum]=getCompRatios(compOb,inte,min_object_size)
%Compartment of interest is stored in channel 1, other compartments in 2
%and 3
L=bwlabel(bwpropfilt(compOb(:,:,1),'Area',[min_object_size+1,Inf]));
ratios=zeros(max(L(:)),4);
%Get textural features of compartment segmentation
g=graycomatrix(inte);
s=graycoprops(g,'all');
s=struct2table(s);
stats=regionprops(L,'area','convexarea');
compNum=max(L(:));
%For all objects
for i=1:compNum
    %Object of interest
    comp=logical(L==i);
    %Other compartments
    CCO=compOb(:,:,2:3);

    %Convex hull of object
    ch=bwconvhull(comp);
    CCO(~repmat(ch,[1,1,2]))=0;

    CCOsum=squeeze(sum(sum(CCO)));

    %Convexity of compartment
    ratios(i,1)=stats(i).Area/stats(i).ConvexArea;
    %Ratio of compartment 2 to compartment 1
    ratios(i,2)=CCOsum(1)/stats(i).Area;
    %Ratio of compartment 3 to compartment 1
    ratios(i,3)=CCOsum(2)/stats(i).Area;
    %Area of compartment of interest
    ratios(i,4)=stats(i).Area;
end
