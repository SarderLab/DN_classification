function v=feature_extraction_inner(q,segmented_gloms,image_dir,min_object_size)
%%Full commentary of the following analysis is available in the version of
%%this code used for human analysis
v=zeros(1,218);
nucpixradius=2;
composite=imread(fullfile(segmented_gloms(q).folder,segmented_gloms(q).name))>0;
I=imread(fullfile(image_dir(q).folder,image_dir(q).name));

mes_mask=composite(:,:,1);
white_mask=composite(:,:,2);
nuc_mask=composite(:,:,3);

boundary_mask=mes_mask|white_mask|nuc_mask;


gdist=bwdist(~boundary_mask);
gdist=(-1*(gdist))+max(gdist(:));
gdist(~boundary_mask)=0;

% Determine glomerular area
gArea=sum(boundary_mask(:));
 % Determine glomerular boundary
gOutline=bwperim(boundary_mask);

% Determine glomerular centroid
[r,c]=find(boundary_mask);
rMean=round(mean(r));
cMean=round(mean(c));

% Get precursor mask from color deconvolution
[a,b,~]=colour_deconvolution(I,'H PAS');

 mesInt=1-im2double(b);

    % Find mesangial segmentation on PAS-stain deconvolved channel using Otsu's
    % method
    mes=imbinarize(mesInt,adaptthresh(mesInt));
mes(~boundary_mask)=0;
mes=bwareaopen(mes,min_object_size);
mdt=bwdist(~mes);

% values are modified as compared to human analysis to match sizes
% consistent with mouse images
m_ext1=mdt>0&mdt<=5;
m_ext2=mdt>5&mdt<=10;
m_ext3=mdt>10&mdt<100;

edges=[1:1:40,200];
N1=histcounts(mdt(mdt(:)>0),edges);
ldt=bwdist(~white_mask);
edges=[1:1:40,200];
N2=histcounts(ldt(ldt(:)>0),edges);

ndt=bwdist(~nuc_mask);
edges=[1:1:20,200];
N3=histcounts(ndt(ndt(:)>0),edges);

edges=[2:10:300,2000];
N4=histcounts(gdist(gdist(:)>0),edges);

% Create grayscale representation of image to determine textural features

grayIm=rgb2gray(I);
grayIm(~mes_mask)=NaN;

%    Determine textural and compartment containment features
[ratiosM,s1,mes_num]=getCompRatios(composite,grayIm,min_object_size);

% Re-orient the segmentation channels so that the function 'getCompRatios'
% knows which segmentation is the primary compartment to be examined
composite=cat(3,white_mask,mes_mask,nuc_mask);
composite(~repmat(boundary_mask,[1,1,3]))=0;
% Repeat the steps above for luminal compartments
grayIm=rgb2gray(I);
grayIm(~white_mask)=NaN;

[ratiosL,s2,lum_num]=getCompRatios(composite,grayIm,min_object_size);

% Re-orient the segmentation channels so that the function 'getNucRatios'
% knows which segmentation is the primary compartment to be examined
composite=cat(3,nuc_mask,white_mask,mes_mask);
composite(~repmat(boundary_mask,[1,1,3]))=0;
grayIm=rgb2gray(I);
grayIm(~nuc_mask)=NaN;
% Get nuclear ratios
[ratiosN,s3,nuc_num]=getNucRatios(composite,nucpixradius,grayIm);

% Get distance features between the glomerular periphery, center, and
% between compartments
distsN=getCompDists(nuc_mask,gOutline,[rMean,cMean]);
distsM=getCompDists(mes_mask,gOutline,[rMean,cMean]);
distsL=getCompDists(white_mask,gOutline,[rMean,cMean]);


%Calculate lumen compartmentalization features
v(1,1:3)=mean(ratiosL(:,1:3));
v(1,4)=sum(ratiosL(:,4));
v(1,5)=mean(ratiosL(:,4));
v(1,6)=median(ratiosL(:,4));

%Unpack luminal texture features
v(1,7:10)=[s1(1,1).Contrast,s1(1,2).Correlation,s1(1,3).Energy,s1(1,4).Homogeneity];

%Calculate mesangial compartmentalization features

v(1,11:13)=mean(ratiosM(:,1:3));
v(1,12)=mean(ratiosM(:,2));
v(1,13)=mean(ratiosM(:,3));
v(1,14)=sum(ratiosM(:,4));
v(1,15)=mean(ratiosM(:,4));
v(1,16)=median(ratiosM(:,4));

v(1,17:20)=[s2(1,1).Contrast,s2(1,2).Correlation,s2(1,3).Energy,s2(1,4).Homogeneity];

%Calculate nuclear comparmentalization features
v(1,21)=mean(ratiosN(:,1));
v(1,22)=mean(ratiosN(:,2));
v(1,23)=mean(ratiosN(:,3));

v(1,24)=sum(ratiosN(:,4));
v(1,25)=mean(ratiosN(:,4));
v(1,26)=mode(ratiosN(:,4));

v(1,27:30)=[s3(1,1).Contrast,s3(1,2).Correlation,s3(1,3).Energy,s3(1,4).Homogeneity];

%Calculate luminal distance features
v(1,31:37)=mean(distsL);

%Calculate mesangial distance features
v(1,38:44)=mean(distsM);
%Calculate nuclear distance features 
v(1,45:51)=mean(distsN);

%Glomerular area
v(1,52)=gArea;
v(1,53)=mes_num;
v(1,54)=lum_num;
v(1,55)=nuc_num;
v(1,56)=sum(sum(m_ext1));
v(1,57)=sum(sum(m_ext2));
v(1,58)=sum(sum(m_ext3));
if sum(sum(m_ext2))==0
    v(1,59)=0;
else
    v(1,59)=max(max(mdt(m_ext2)));
end
v(1,60)=max(max(bwlabel(m_ext1)));
v(1,61)=max(max(bwlabel(m_ext2)));

v(1,62)=mean(mean(mdt(m_ext1>0)));
v(1,63)=mean(mean(mdt(m_ext2>0)));


v(1,64)=median(mdt(m_ext1>0));
v(1,65)=median(mdt(m_ext2>0));

%%%
stats=regionprops(m_ext1,'Area');
v(1,66)=mean([stats.Area]);
v(1,67)=median([stats.Area]);
v(1,68)=max([stats.Area]);


stats2=regionprops(m_ext2,'Area');
v(1,69)=mean([stats2.Area]);
v(1,70)=median([stats2.Area]);

v(1,71:71+(length(N1)-1))=N1;



v(1,111:111+(length(N2)-1))=N2;

v(1,151:151+(length(N3)-1))=N3;

v(1,171:171+(length(N4)-1))=N4;

[d1,d2,z]=size(I);

mes_int=im2double(I);
mes_int(~repmat(mes,[1,1,3]))=NaN;
mes_int=reshape(mes_int,[d1*d2,3]);

lum_int=im2double(I);
lum_int(~repmat(white_mask,[1,1,3]))=NaN;
lum_int=reshape(lum_int,[d1*d2,3]);

nuc_int=im2double(I);
nuc_int(~repmat(nuc_mask,[1,1,3]))=NaN;
nuc_int=reshape(nuc_int,[d1*d2,3]);

v(1,201)=mean(mes_int(:,1),'omitnan');
v(1,202)=mean(mes_int(:,2),'omitnan');
v(1,203)=mean(mes_int(:,3),'omitnan');

v(1,204)=std(mes_int(:,1),[],'omitnan');
v(1,205)=std(mes_int(:,2),[],'omitnan');
v(1,206)=std(mes_int(:,3),[],'omitnan');


v(1,207)=mean(lum_int(:,1),'omitnan');
v(1,208)=mean(lum_int(:,2),'omitnan');
v(1,209)=mean(lum_int(:,3),'omitnan');
v(1,210)=std(lum_int(:,1),[],'omitnan');
v(1,211)=std(lum_int(:,2),[],'omitnan');
v(1,212)=std(lum_int(:,3),[],'omitnan');

v(1,213)=mean(nuc_int(:,1),'omitnan');
v(1,214)=mean(nuc_int(:,2),'omitnan');
v(1,215)=mean(nuc_int(:,3),'omitnan');
v(1,216)=std(nuc_int(:,1),[],'omitnan');
v(1,217)=std(nuc_int(:,2),[],'omitnan');
v(1,218)=std(nuc_int(:,3),[],'omitnan');

end