%function recommenderSystem()
start = tic;
%Read in the data from *.dat files into useful matrices and cell
%arrays, given the file locations and genre list
path = strcat(['..',filesep,'RSdata',filesep]);
genres = {'Action','Adventure','Animation','Children''s',...
    'Comedy','Crime','Documentary','Drama','Fantasy',...
    'Film-Noir','Horror','Musical','Mystery','Romance',...
    'Sci-Fi','Thriller','War','Western','M','F'}';
numDemos = 2;
data = DataCache(path,genres,numDemos);

%creates a numUsers x numMovies matrix to store all ratings
maxs = max(data.train, [], 1);
hugeAssMatt = zeros(maxs(1), maxs(2));
for sample = 1:1:(size(data.train,1))
    user = data.train(sample,1);
    movie = data.train(sample,2);
    rating = data.train(sample,3);
    hugeAssMatt(user, movie) = rating;
end

%avgReviewsUser: vector, average each user rates all movies
%avgReviewsMovie: vector, average rating each movie receives
numReviewsUser = sum(hugeAssMatt ~= 0, 2);
numReviewsMovie = sum(hugeAssMatt ~= 0, 1);
avgReviewsUser = sum(hugeAssMatt, 2) ./ numReviewsUser;
avgReviewsMovie = sum(hugeAssMatt, 1) ./ numReviewsMovie;

%deals with case where a user or movie has no ratings
avgReviewsUser(isnan(avgReviewsUser)) = 0;
avgReviewsMovie(isnan(avgReviewsMovie)) = 0;

%avgReviewsOverall: average rating across all users and movies
%movieOffsets: how much each movie average differs from the total average
%ouserOffsets: how much each user average differs from the total average
totalReviews = nnz(hugeAssMatt);
avgReviewsOverall = sum(sum(hugeAssMatt)) / totalReviews;
movieOffsets = avgReviewsMovie - avgReviewsOverall;
userOffsets = avgReviewsUser - avgReviewsOverall;

%pretraining for feature matrices
data = pretrainRS(data);

%calculates offset of each users average rating of movies of a particular
%genre compared to their average rating across all movies
%store this data in data.userMat
%the last two columns are latent features for gender
data.userMat = data.userMat - repmat(avgReviewsUser,1,20);
data.userMat(:,19:20) = 0;

%learn feature matrices
data = gradientDescent(data, avgReviewsOverall, userOffsets, movieOffsets);

%Assign ratings to the test data based on learned parameters
assignments = assignRS(data,avgReviewsOverall,userOffsets,movieOffsets);
%Output recommendations for test set in a *.csv file
csvwrite(strcat(['..',filesep,'results.csv']),assignments);
toc(start)
%end