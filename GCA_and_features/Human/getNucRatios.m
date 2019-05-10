function [ratios,s,compNum]=getNucRatios(compOb,nucpixradius,graynuclei)
L=bwlabel(compOb(:,:,1));
ratios=zeros(max(L(:)),4);
g=graycomatrix(graynuclei);
s=graycoprops(g,'all');
s=struct2table(s);
compNum=max(L(:));
for i=1:compNum
    %Get nuclear compartment
    comp=logical(L==i);
    %Find pixels surrounding each nucleus
    compOutline=bwperim(bwmorph(comp,'dilate',nucpixradius));
    %Eliminate pixels that aren't on the nuclear border
    CCO=compOb;
    CCO(~repmat(compOutline,[1,1,3]))=0;
    
    %Quantify compartment values at the nuclear boundary
    ratios(i,1)=sum(sum(CCO(:,:,2)))/sum(sum(compOutline));
    ratios(i,2)=sum(sum(CCO(:,:,3)))/sum(sum(compOutline));
    %Length of nuclear boundary
    ratios(i,3)=sum(sum(compOutline));
    % Nuclear area
    ratios(i,4)=sum(sum(comp));
end
