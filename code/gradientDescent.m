function data = gradientDescent(data)
gamma = .001;
convergeVal = .00005;
userChange = 1;
movieChange = 1;
i = 0;
%Iterate through all training examples, updating learned user preferences
% while max(abs(userChange)) > convergeVal || ...
%         max(abs(movieChange)) > convergeVal
for epoch = 1:1:5
    for sample = 1:1:size(data.train,1)
        %i = i + 1
        %sample = round(rand*(size(data.train,1)-1))+1;
        %Get the current sample information
        user = data.train(sample,1);
        movie = data.train(sample,2);
        rating = data.train(sample,3);
        numGenres = sum(data.movieMat(movie,:)');
        error = rating - sum(data.userMat(user,:).*data.movieMat(movie,:))/...
            numGenres;
        userMatGrad = -error*data.movieMat(movie,:);
        movieMatGrad = -error*data.userMat(user,:);
        userChange = -gamma*userMatGrad;
        data.userMat(user,:) = data.userMat(user,:) + userChange;
        movieChange = -gamma*movieMatGrad;
        data.movieMat(movie,:) = data.movieMat(movie,:) + movieChange;
    end
    error
end
end