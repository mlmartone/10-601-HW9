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
                'Sci-Fi','Thriller','War','Western','M','F'}';
    numDemos = 2;
    data = DataCache(path,genres,numDemos);
else
    %If the data isn't reinitialized, reset learned parameters
    data.userMat = zeros(size(data.userMat));
end

%creates a numUsers x numMoves matrix to store each users rating
%for each movie
maxs = max(data.train, [], 1)
hugeAssMatt = zeros(maxs(1), maxs(2));
for sample = 1:1:(size(data.train,1))
    user = data.train(sample,1);
    movie = data.train(sample,2);
    rating = data.train(sample,3);
    hugeAssMatt(user, movie) = rating;
end

%avgReviewsUser: vector storing average rating given by each user
%avgReviewsMovie: vector storing average rating for each movie
numReviewsUser = sum(hugeAssMatt ~= 0, 2);
numReviewsMovie = sum(hugeAssMatt ~= 0, 1);
avgReviewsUser = sum(hugeAssMatt, 2) ./ numReviewsUser;
avgReviewsMovie = sum(hugeAssMatt, 1) ./ numReviewsMovie;

%avgReviewsOverall: average rating across all movies
%movieOffsets: how much each movie average differs from the overall
%userOffsets: how much each user average differs from the overall
totalReviews = nnz(hugeAssMatt);
avgReviewsOverall = sum(sum(hugeAssMatt)) / totalReviews;
movieOffsets = avgReviewsMovie - avgReviewsOverall;
userOffsets = avgReviewsUser - avgReviewsOverall;

%calculates the average rating per genre for each userID
%stores this data in data.userMat
data = trainRS(data);

%userMat will now store how much a users rating for a particular genre
%differs from their average rating across all movies
data.userMat = data.userMat - repmat(avgReviewsUser, 1, 20);

%we should probably delete this, its just so we ignore the M F columns
data.userMat(:,19:20) = 0;


data = gradientDescent(data, avgReviewsOverall, userOffsets, movieOffsets);

%Assign ratings to the test data based on learned parameters
assignments = assignRS(data,avgReviewsOverall,userOffsets,movieOffsets);
if(output)
    %Output recommendations for test set in a *.csv file
    csvwrite(strcat(['..',filesep,'results.csv']),assignments);
end
toc(start)
%end
