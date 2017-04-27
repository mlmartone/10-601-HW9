classdef DataCache
    %A class that stores the data for a recommender system
    
    properties
        movies;
        users;
        train;
        test;
        movieMat;
        userMat;
        genres;
    end
    
    methods
        %Creates a DataCache object with the data a the specified path
        function data = DataCache(path,genres)
            data.genres = genres;
            %Read in and format the movies file into a cell array
            %data.movies is in the format (movie_id,genres)
            data.movies = readtable(strcat([path,'movies.dat']),...
                'Delimiter',':','ReadVariableNames',0);
            data.movies = table2cell(data.movies(:,[1,3]));
            for i = 1:1:size(data.movies,1)
                data.movies{i,2} = strsplit(data.movies{i,2},'|');
            end
            %Read in and format the users file into a cell array
            %data.users is in the format (user_id,sex,age_group,occupation)
            data.users = readtable(strcat([path,'users.dat']),...
                'Delimiter',':','ReadVariableNames',0);
            data.users = table2cell(data.users(:,[1,3,5,7]));
            %Read in and format the training data into a matrix
            %data.train is in the format (user_id,movie_id,rating)
            data.train = readtable(strcat([path,...
                'training_rating.dat']),'Delimiter',':',...
                'ReadVariableNames',0);
            data.train = cell2mat(table2cell(data.train(:,[1,3,5])));
            data.train = data.preprocess(data.train);
            %Read in and format the test data into a matrix
            %data.test is in the format (user_id,movie_id)
            data.test = readtable(strcat([path,'testing.dat']),...
                'Delimiter',' ','ReadVariableNames',0);
            data.test = cell2mat(table2cell(data.test));
            %Generate an easier to work with movieMat matrix that expands
            %the movies to be in any combination of 18 given genres
            data.movieMat = data.expandMovies(data.movies,data.genres);
            %Generate an easier to work with userMat matrix that expands
            %the users to rate any combination of 18 given genres
            %NOTE: THIS MATRIX IS EMPTY AND FILLED IN LATER
            data.userMat = zeros(size(data.users,1),size(data.genres,1));
        end
    end
    
    methods(Static)
        %Pre-Processes the given data by removing rows with NaN values
        function dataMat = preprocess(dataMat)
            for i = 1:1:size(dataMat,2)
                dataMat = dataMat(~isnan(dataMat(:,i)),:);
            end
        end
        
        %Expands the movies cell array to an easier to work with matrix of
        %genres, indexed by (movie_id,genre)
        function movieMat = expandMovies(movies,genres)
            movieMat = zeros(size(movies,1),size(genres,1));
            for movie = 1:1:size(movies,1)
                for genre = 1:1:size(genres,1)
                    movieMat(movie,genre) = any(any(strcmp(...
                        movies{movie,2},genres(genre))));
                end
            end
        end
    end
end

