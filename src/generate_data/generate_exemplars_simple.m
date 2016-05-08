function [ outs ] = generate_exemplars_simple( prototype, folder, category ,n )
%GENERATE_EXEMPLARS_SIMPLE Generates 'n' exemplars by simple warping.
if nargin<4
    n = 40;
end
outs = prototype;
for num=1:n
    num
    out = random_warp(prototype);
    imwrite(out,strcat(folder,'/imgs/',category,num2str(num),'.png'),'png');
    outs = cat(3,outs,out);
end
save(strcat(folder,'/',category),'outs');
end

