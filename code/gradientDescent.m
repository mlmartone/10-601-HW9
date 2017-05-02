function data = gradientDescent(data, avg, userB, movieB)

%training paramters
gamma = .007;
numEpochs = 50;
lambda = 0.1;

error = zeros(size(data.train,1),1);

%Iterate through training data numEpochs times
%updating userMat and movieMat parameters using gradient descent
for epoch = 1:1:numEpochs
    
    %training on all data for autolab handin
    %for parameter tuning we trained on 1/10 of the training data
    for sample = 1:1:size(data.train,1)
        
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
    
    
    %commented out for Autolab handin
    %we used the other 9/10 of the training data that we didn't train on
    %to calculate a test error to be used in parameter tuning
    %for test = 100000:1:size(data.train,1)
    %    user = data.train(test,1);
    %    movie = data.train(test,2);
    %    rating = data.train(test,3);
    %    error(test - 100000 + 1) = rating - ...
    %        sum(data.userMat(user,:).*data.movieMat(movie,:));
    %end
    %error(epoch+1) = rms(error);
    %plot(0:1:numEpochs,totalError);
    %pause(.01);
end

end
