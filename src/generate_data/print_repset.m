function [ mapped ] = print_repset( folder )
%PRINT_REPSET Prints the representative set
load(strcat(folder,'/rep_set'),'mapped');
csvwrite(strcat(folder,'/rep_set.dat'),mapped);
end

