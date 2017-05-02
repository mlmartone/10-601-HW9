function data = gradientDescent(data, avg, userB, movieB)
gamma = .007;
numEpochs = 25;
lambda = 0.1;
testAmt = 1;
error = zeros(size(data.train,1),1);
validationError = zeros(size(data.train,1) - testAmt, 1);
totalError = zeros(numEpochs+1,1);
%for movie = 1:1:size(data.movieMat,1)
%    data.movieMat(movie,:) = data.movieMat(movie,:)/...
%        sum(data.movieMat(movie,:)');
%end
for sample = 1:1:size(data.train,1)
    %Get the current sample information
    user = data.train(sample,1);
    movie = data.train(sample,2);
    rating = data.train(sample,3);
    estimate =  avg +  userB(user) + movieB(movie) + ...
        sum(data.userMat(user,:).*data.movieMat(movie,:));
    error(sample) = rating - estimate;
end
totalError(1) = rms(error);
%Iterate through all training examples, updating with gradient descent
size(data.train,1)
for epoch = 1:1:numEpochs
    epoch
    for sample = 1:1:size(data.train,1)
        %Get the current sample information
        user = data.train(sample,1);
        movie = data.train(sample,2);
        rating = data.train(sample,3);
        estimate =  avg +  userB(user) + movieB(movie) + ...
            sum(data.userMat(user,:).*data.movieMat(movie,:));
        error(sample) = rating - estimate;
        userMatGrad = -error(sample)*data.movieMat(movie,:) + ...
            lambda*data.userMat(movie,:);
        movieMatGrad = -error(sample)*data.userMat(user,:) + ...
            lambda*data.movieMat(movie,:);
        userChange = -gamma*userMatGrad;
        data.userMat(user,:) = data.userMat(user,:) + userChange;
        movieChange = -gamma*movieMatGrad;
        data.movieMat(movie,:) = data.movieMat(movie,:) + movieChange;
    end
    %gamma = max(gamma* .6,.00001);
    %for sample = testAmt:1:size(data.train,1)
    %    user = data.train(sample,1);
    %    movie = data.train(sample,2);
    %    rating = data.train(sample,3);
    %    estimate =  avg +  userB(user) + movieB(movie) + ...
    %        sum(data.userMat(user,:).*data.movieMat(movie,:));
    %    validationError(sample - testAmt + 1) = rating - estimate;
    %end
    %totalError(epoch+1) = rms(validationError);
    error(epoch+1)
    %plot(0:1:numEpochs,totalError);
    %pause(.01)
end

end