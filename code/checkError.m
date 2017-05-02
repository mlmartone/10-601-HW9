%function rmse = checkError(data,assignments)
%Read in and format the training data into a matrix
%data.train is in the format (user_id,movie_id,rating)
trueRatings = readtable('ratingsFULL.dat','Delimiter',':',...
    'ReadVariableNames',0);
trueRatings = cell2mat(table2cell(trueRatings(:,[1,3,5])));
error = zeros(size(assignments));
for sample = 1:1:size(data.test,1)
    %Get the current sample information
    user = data.test(sample,1);
    movie = data.test(sample,2);
    rating = trueRatings(trueRatings(:,1) == user,:);
    rating = rating(rating(:,2) == movie,:);
    if(size(rating,1) ~= 0)
        error(sample) = (rating(3) - assignments(sample))^2;
        sample
    end
end
rmse = rms(error);
%end