%Trains a recommender system given movie and user data
function data = trainRS(data)
%Set up a matrix of reviews by genre to normalize ratings
numReviews = zeros(size(data.userMat,1),size(data.genres,1));
%Iterate through all training examples, updating learned user preferences
for sample = 1:1:size(data.train,1)
    %Get the current sample information
    user = data.train(sample,1);
    movie = data.train(sample,2);
    rating = data.train(sample,3);
    %Update the number of reviews and total review for the user based on
    %the current sample
    numReviews(user,:) = numReviews(user,:) + data.movieMat(movie,:);
    data.userMat(user,:) = data.userMat(user,:) + ...
        rating*data.movieMat(movie,:);
end
%Normalize ratings of a given user by genre
numReviews(find(numReviews == 0)) = 1;
avgReview = sum(data.userMat')'./sum(numReviews')';
data.userMat = data.userMat./numReviews;
%If a user has not rated any movies of a given genre, assign them the
%average review of that user
data.userMat(find(data.userMat == 0)) = avgReview(user);
end