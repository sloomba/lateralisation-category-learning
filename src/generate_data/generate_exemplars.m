function [ outs ] = generate_exemplars( prototype, folder, n, blk_sz, num_ev, sigma)
%DON'T USE THIS. IF YOU WISH TO USE THIS, SOME CHANGES WOULD BE REQUIRED.
%GENERATE_EXEMPLARS Generates 'n' exemplar images for the given RGB/
%Grayscale square prototype image of max size 1024x1024, min size 64X64,
%(preferably powers of 2, else truncated to highest power of 2 just smaller 
%than the image dimension). Block size is given by 'blk_sz', number of
%eigenvalues used is given by 'num_ev' and std dev of Gaussian spread given
%by 'sigma'.
if nargin<6
    sigma = 100;
    if nargin<5
        num_ev = 4;
        if nargin<4
            blk_sz=32;
            if nargin<3;
                n = 40;
            end
        end
    end
end
cut=0;
for i=6:10
    if size(prototype,1)>=2^i && size(prototype,1)<2^(i+1)
        cut = 2^i;
        break;
    end
end
if cut==0
    display('Image specs are not appropriate.');
    return;
else
    prototype = prototype(1:cut,1:cut); %truncate image
end
prototype = im2bw(prototype,0.5); %convert image to BW-double.
num_subblocks = cut/blk_sz;
subblocks = [];
for i=0:num_subblocks-1
    for j=0:num_subblocks-1
        subblock = prototype(i*blk_sz+1:(i+1)*blk_sz,j*blk_sz+1:(j+1)*blk_sz);
        subblock = reshape(subblock',1,blk_sz^2); %flatten subblock to vector
        subblocks = cat(1,subblocks,subblock);
    end
end
[u,s,v] = svd(subblocks'); %SVD analysis
s_orig = zeros(1,num_ev);
for i=1:num_ev
    s_orig(i)=s(i,i);
end
for i=num_ev+1:min(size(s))
    s(i,i)=0;
end
outs = [];
h = fspecial('gaussian',[21 21], 100);
for num=0:n
    out = transpose(u*s*v');
    outmats = [];
    k = 1;
    for i=0:num_subblocks-1
        outmat = [];
        for j=0:num_subblocks-1
            outmat = cat(2,outmat,transpose(reshape(out(k,:),blk_sz,blk_sz))); %unflatten vector subblock
            k = k+1;
        end
        outmats = cat(1,outmats,outmat);
    end
    outmats = imfilter(outmats,h);
    imwrite(outmats,strcat(folder,'/',folder,'_',num2str(num),'.jpg'),'jpg');
    for i=1:num_ev
        s(i,i)=normrnd(s_orig(i),sigma);
    end
    outs = cat(3,outs,outmats);
end
save(strcat(folder,'/',folder),'outs');
end

