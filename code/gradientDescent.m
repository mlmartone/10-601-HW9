function data = gradientDescent(data, avg, userB, movieB)

%training paramters
gamma = .007;
numEpochs = 50;
lambda = 0.1;
testAmt = 10000;

error = zeros(size(data.train,1),1);
validationError = zeros(size(data.train,1) - testAmt, 1);
totalError = zeros(numEpochs+1,1);

for sample = 1:1:size(data.train,1)
    user = data.train(sample,1);
    movie = data.train(sample,2);
    rating = data.train(sample,3);
    estimate =  avg +  userB(user) + movieB(movie) + ...
        sum(data.userMat(user,:).*data.movieMat(movie,:));
    error(sample) = rating - estimate;
end
totalError(1) = rms(error);

%Iterate through training data numEpochs times
%updating userMat and movieMat parameters using gradient descent
for epoch = 1:1:numEpochs
    
    %training on all data for autolab handin
    %for parameter tuning we trained on 1/10 of the training data
    for sample = 1:1:size(data.train,1) %testAmt
        
        %Get the current sample information
        user = data.train(sample,1);
        movie = data.train(sample,2);
        rating = data.train(sample,3);
        
        %predict the rating for the user and movie
        %bias the prediction using overall average and user/movie offset
        estimate =  avg +  userB(user) + movieB(movie) + ...
            sum(data.userMat(user,:).*data.movieMat(movie,:));
        
        %calculate the error and gradient using L2 regularization
        error(sample) = rating - estimate;
        userMatGrad = -error(sample)*data.movieMat(movie,:) + ...
            lambda*data.userMat(movie,:);
        movieMatGrad = -error(sample)*data.userMat(user,:) + ...
            lambda*data.movieMat(movie,:);
        
        %update feature matrices
        userChange = -gamma*userMatGrad;
        data.userMat(user,:) = data.userMat(user,:) + userChange;
        movieChange = -gamma*movieMatGrad;
        data.movieMat(movie,:) = data.movieMat(movie,:) + movieChange;
    end
    
    %reduce descent rate each epoch
    gamma = max(gamma*.95,.000001);
    
    
    %we used the other 9/10 of the training data that we didn't train on
    %to calculate a test error to be used in parameter tuning
    for test = testAmt:1:size(data.train,1)
        user = data.train(test,1);
        movie = data.train(test,2);
        rating = data.train(test,3);
        validationError(test - testAmt + 1) = rating - ...
            sum(data.userMat(user,:).*data.movieMat(movie,:));
    end
    totalError(epoch+1) = rms(error); %validationError);
    plot(0:1:numEpochs,totalError);
    pause(.01);
end

end
