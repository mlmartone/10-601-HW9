%function recommenderSystem()
path = strcat(['..',filesep,'RSdata',filesep]);
data = DataCache(path);
output = rand([size(data.test,1),1])*5;
csvwrite('results',output);
%end