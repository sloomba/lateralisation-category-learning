function [ mapped ] = compare_categories( folder )
%COMPARE_CATEGORIES Compares the two category stimuli by doing PCA
%analysis.
load(strcat(folder,'/a'),'outs');
catA = outs;
load(strcat(folder,'/b'),'outs');
catB = outs;
image_set = [];
sz = size(catA,1);
nm = size(catA,3);
for i=1:nm
    image_set = cat(1,image_set,reshape(catA(:,:,i)',1,sz^2));
end
for i=1:nm
    image_set = cat(1,image_set,reshape(catB(:,:,i)',1,sz^2));
end
image_set = double(image_set/255);
mapped = pca(image_set,8); %PCA analysis
save(strcat(folder,'/rep_set'),'mapped');
scatter(mapped(1:nm,1),mapped(1:nm,2),'r');
for i=1:nm
    text(mapped(i,1)+0.5,mapped(i,2)+0.5,num2str(i));
end
hold all;
scatter(mapped(nm+1:end,1),mapped(nm+1:end,2),'b');
for i=1:nm
    text(mapped(i+nm,1)+0.5,mapped(i+nm,2)+0.5,num2str(i));
end
hold off;
title('Scatter plot of stimuli dataset in top 2 PCA dimensions');
xlabel('Dimension 1');
ylabel('Dimension 2');
legend('a','b');
print(strcat(folder,'/PCA2'),'-dpng')
end

