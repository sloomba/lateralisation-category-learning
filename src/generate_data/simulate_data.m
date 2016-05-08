%Main script. Make appropriate changes here, and run this script.
folder_name = 'kanji1_kanji2'; %change to destination folder name
protoA_name = 'prototypes/kanji1.png'; %change to prototype A name
protoB_name = 'prototypes/kanji2.png'; %change to prototype B name
%mkdir(folder_name)
%mkdir(strcat(folder_name,'/imgs'));
%display('Generating exemplars for Category A...');
%generate_exemplars_simple(imread(protoA_name),folder_name,'a');
%display('Generating exemplars for Category B...');
%generate_exemplars_simple(imread(protoB_name),folder_name,'b');
%compare_categories(folder_name);
print_repset(folder_name);