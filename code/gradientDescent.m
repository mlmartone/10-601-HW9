function data = gradientDescent(data)
gamma = .0015;
numEpochs = 200;
lambda = 0;%-.02;
testAmt = size(data.train,1);
error = zeros(size(data.train,1),1);
validationError = zeros(size(data.train,1) - testAmt, 1);
totalError = zeros(numEpochs+1,1);
for movie = 1:1:size(data.movieMat,1)
    data.movieMat(movie,1:size(data.genres,1)-data.numDemos) = ...
        data.movieMat(movie,1:size(data.genres,1)-data.numDemos)/...
        sum(data.movieMat(movie,1:size(data.genres,1)-data.numDemos)');
end
for sample = 1:1:size(data.train,1)
    %Get the current sample information
    user = data.train(sample,1);
    movie = data.train(sample,2);
    rating = data.train(sample,3);
    error(sample) = rating - ...
        sum(data.userMat(user,:).*data.movieMat(movie,:));
end
totalError(1) = rms(error);
%Iterate through all training examples, updating with gradient descent
for epoch = 1:1:numEpochs
    for sample = 1:1:testAmt %size(data.train,1)
        %Get the current sample information
        user = data.train(sample,1);
        movie = data.train(sample,2);
        rating = data.train(sample,3);
        error(sample) = rating - ...
            sum(data.userMat(user,:).*data.movieMat(movie,:));
        userMatGrad = -error(sample)*data.movieMat(movie,:) + lambda*data.userMat(movie,:);
        movieMatGrad = -error(sample)*data.userMat(user,:) + ...
            lambda*data.movieMat(movie,:);
        userChange = -gamma*userMatGrad;
        data.userMat(user,:) = data.userMat(user,:) + userChange;
        movieChange = -gamma*movieMatGrad;
        data.movieMat(movie,:) = data.movieMat(movie,:) + movieChange;
    end
    gamma = max(gamma*.95,.000001);
    for sample = testAmt:1:size(data.train,1)
        user = data.train(sample,1);
        movie = data.train(sample,2);
        rating = data.train(sample,3);
        validationError(sample - testAmt + 1) = rating - ...
            sum(data.userMat(user,:).*data.movieMat(movie,:));
    end
    totalError(epoch+1) = rms(error);%validationError);
    plot(0:1:numEpochs,totalError);
    pause(.01);
end

end