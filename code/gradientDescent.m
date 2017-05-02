function data = gradientDescent(data)
gamma = .002;
numEpochs = 250;
lambda = 0;
error = zeros(size(data.train,1),1);
totalError = zeros(numEpochs+1,1);
for movie = 1:1:size(data.movieMat,1)
    data.movieMat(movie,:) = data.movieMat(movie,:)/...
        sum(data.movieMat(movie,:)');
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
    for sample = 1:1:size(data.train,1)
        %Get the current sample information
        user = data.train(sample,1);
        movie = data.train(sample,2);
        rating = data.train(sample,3);
        error(sample) = rating - ...
            sum(data.userMat(user,:).*data.movieMat(movie,:));
        userMatGrad = -error(sample)*data.movieMat(movie,:);
        movieMatGrad = -error(sample)*data.userMat(user,:) + ...
            lambda*data.movieMat(movie,:);
        userChange = -gamma*userMatGrad;
        data.userMat(user,:) = data.userMat(user,:) + userChange;
        movieChange = -gamma*movieMatGrad;
        data.movieMat(movie,:) = data.movieMat(movie,:) + movieChange;
    end
    gamma = max(gamma*.99,.0005);
    totalError(epoch+1) = rms(error);
    plot(0:1:numEpochs,totalError);
    pause(.01)
end

end