%function recommenderSystem()
%Set these when running the code to suppress certain behaviors by setting
%them to 0
start = tic;
readData = 1;
output = 1;
if(readData)
    %Read in the data from *.dat files into useful matrices and cell
    %arrays, given the file locations and genre list
    path = strcat(['..',filesep,'RSdata',filesep]);
    genres = {'Action','Adventure','Animation','Children''s',...
                'Comedy','Crime','Documentary','Drama','Fantasy',...
                'Film-Noir','Horror','Musical','Mystery','Romance',...
                'Sci-Fi','Thriller','War','Western','Male','Female'}';
    data = DataCache(path,genres);
else
    %If the data isn't reinitialized, reset learned parameters
    data.userMat = zeros(size(data.userMat));
end
%Train on the given data
data = trainRS(data);
data = gradientDescent(data);
%Assign ratings to the test data based on learned parameters
assignments = assignRS(data);
if(output)
    %Output recommendations for test set in a *.csv file
    csvwrite(strcat(['..',filesep,'results.csv']),assignments);
end
toc(start)
%end