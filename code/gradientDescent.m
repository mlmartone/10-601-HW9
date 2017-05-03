function data = gradientDescent(data, avg, userB, movieB)
gamma = .007;
numEpochs = 100;
lambda = 0.1;

%since we are using the entire dataset to train for Autolab submission
trainAmt = 1;

error = zeros(size(data.train,1),1);
validationError = zeros(size(data.train,1) - trainAmt, 1);
totalError = zeros(numEpochs+1,1);


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
for epoch = 1:1:numEpochs

    for sample = 1:1:size(data.train,1)
        %Get the current sample information
        user = data.train(sample,1);
        movie = data.train(sample,2);
        rating = data.train(sample,3);
        
        %Predict the rating for the user and movie
        %Bias the prediction using overall average and user/movie offset
        estimate =  avg +  userB(user) + movieB(movie) + ...
            sum(data.userMat(user,:).*data.movieMat(movie,:));
        
        %Calculate the error and gradient using L2 regularization
        error(sample) = rating - estimate;
        userMatGrad = -error(sample)*data.movieMat(movie,:) + ...
            lambda*data.userMat(movie,:);
        movieMatGrad = -error(sample)*data.userMat(user,:) + ...
            lambda*data.movieMat(movie,:);
        
        %Update feature matrices
        userChange = -gamma*userMatGrad;
        data.userMat(user,:) = data.userMat(user,:) + userChange;
        movieChange = -gamma*movieMatGrad;
        data.movieMat(movie,:) = data.movieMat(movie,:) + movieChange;
    end
    
    %Reduce descent rate each epoch to some minumum
    gamma = max(gamma*.95,.000001);
    
    %change trainAmt to be number of samples for training
    %remaining are used for testing
    for sample = trainAmt:1:size(data.train,1)
        user = data.train(sample,1);
        movie = data.train(sample,2);
        rating = data.train(sample,3);
        estimate =  avg +  userB(user) + movieB(movie) + ...
            sum(data.userMat(user,:).*data.movieMat(movie,:));
        
        %Calculate validation error
        validationError(sample - trainAmt + 1) = rating - estimate;
    end
    
    %Error Plotting
    totalError(epoch+1) = rms(error);%validationError);
    plot(0:1:numEpochs,totalError);
    pause(.01);
end

end
