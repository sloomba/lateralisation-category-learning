function [ out ] = random_warp( img, n, blk_sz )
%RANDOM_MORPH Creates a random_morphing of a given image, with n control
%points. CAUTION: If spline function not invertible, error can occur: rare.
if nargin<3
    blk_sz = round(size(img,1)/8);
    if nargin<2
        n = round(size(img,1)/2);
    end
end
height = size(img,1);
width = size(img,2);
idx_from = sort(randi(width*height,n,1)); %control point linear indexes
[idx_from_x,idx_from_y] = ind2sub([height,width],idx_from);
perturb = randi([-blk_sz,blk_sz],2,n);
idx_from = cat(1,idx_from_x',idx_from_y');
idx_to = idx_from + perturb;
idx_to(idx_to<1)=1;
idx_to(idx_to(1,1:n)>width)=width;
idx_to(idx_to(2,1:n)>height)=height;
idx_from = cat(2,[1;1],[1;height],[width;height],[width;1],idx_from);
idx_to = cat(2,[1;1],[1;height],[width;height],[width;1],idx_to);
spline = tpaps(idx_from,idx_to);
mapping = round(fnval(spline,cat(1,repmat(1:width,1,height),reshape(repmat(1:height,width,1),1,height*width))));
mapping(mapping<1)=1;
mapping(mapping(1,1:height*width)>width)=width;
mapping(mapping(2,1:height*width)>height)=height;
out = zeros(height,width);
k = 1;
angle = randi(360,1);
img = imrotate(img,angle);
for i=1:height
    for j=1:width
        out(j,i) = img(min(height,mapping(1,k)),min(width,mapping(2,k)));
        k = k+1;
    end
end
h = fspecial('gaussian',[5 5], 2);
out = imfilter(out,h);
out = uint8(out);
end

