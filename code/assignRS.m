%Assigns a rating to each point in the test set and returns the list of
%ratings
function assignments = assignRS(data)
assignments = zeros(size(data.test,1),1);
%Iterate through each test sample and assign it a rating
for sample = 1:1:size(data.test,1)
    %Get the current sample information
    user = data.test(sample,1);
    movie = data.test(sample,2);
    %Find the assignment for the given pair by taking the dot product of
    %the feature vector and the user's rating vector
    assignments(sample) = sum(data.userMat(user,:).*...
        data.movieMat(movie,:));
end
assignments(assignments < 1) = 1;
assignments(assignments > 5) = 5;
end