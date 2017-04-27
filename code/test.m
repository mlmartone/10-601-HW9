% movies = readtable('../RSdata/movies.dat','Delimiter',':','ReadVariableNames',0);
% movies = table2cell(movies(:,[1,3]));
% for i = 1:1:size(movies,1)
%     movies{i,2} = strsplit(movies{i,2},'|');
% end
% users = readtable(strcat([path,'users.dat']),...
%     'Delimiter',':','ReadVariableNames',0);
% users = table2cell(users(:,[1,3,5,7]));
% training = readtable(strcat([path,'training_rating.dat']),...
%     'Delimiter',':','ReadVariableNames',0);
% training = cell2mat(table2cell(training(:,[1,3,5])));
testdat = readtable(strcat([path,...
    'testing.dat']),'Delimiter',' ',...
    'ReadVariableNames',0);
testdat = cell2mat(table2cell(testdat));