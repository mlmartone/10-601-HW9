classdef DataCache
    %A class that stores the data for a recommender system
    
    properties
        movies;
        users;
        train;
        test;
    end
    
    methods
        %Creates a DataCache object with the data a the specified path
        function data = DataCache(path)
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
        end
    end
    
    methods(Static)
        %Pre-Processes the given data by removing rows with NaN values
        function dataMat = preprocess(dataMat)
            for i = 1:1:size(dataMat,2)
                dataMat = dataMat(~isnan(dataMat(:,i)),:);
            end
        end
    end
end

