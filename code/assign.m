function assignments = assign(data)
assignments = zeros(size(data.test,1),1);
for sample = 1:1:size(data.test,1)
    user = data.test(sample,1);
    movie = data.test(sample,2);
    numGenres = sum(data.movieMat(movie,:)');
    assignments(sample) = sum(data.userMat(user,:).*...
        data.movieMat(movie,:))/numGenres;
end
end