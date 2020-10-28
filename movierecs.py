import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data_movie = pd.read_csv("/Users/navyasogi/Desktop/movies.csv")
data_rating = pd.read_csv("/Users/navyasogi/Desktop/ratings.csv")

movie = data_movie.loc[:,{"movieId","title"}]
rating = data_rating.loc[:,{"userId","movieId","rating"}]

data = pd.merge(movie,rating)
data = data.iloc[:1000000,:]


combine_movie_rating = df.dropna(axis = 0, subset = ['title'])
movie_ratingCount = (combine_movie_rating.
     groupby(by = ['title'])['rating'].apply(list).reset_index().
     rename(columns = {'rating': 'ratingVector'})
     [['title', 'ratingVector']]
    )
movie_ratingCount.head()
user_movie_table = data.pivot_table(index = ["movieId"],columns = ["userId"],values = "rating").fillna(0)
user_movie_table.head(10)

user_movie_table_array = np.array(user_movie_table)
flattened_array = user_movie_table_array.flatten()

query_index = np.random.choice(user_movie_table_array.shape[0])
print("Choosen Movie is: ",user_movie_table.index[query_index])

user_movie_table_matrix = csr_matrix(user_movie_table.values)

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = []
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = []
	for i in range(num_neighbors):
		neighbors.append(distances[i][0])
	return neighbors

# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range (len(row1)-1) :
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

num_neighbors = 5
scores = get_neighbors(movie_ratingCount.ratingVector, movie_ratingCount.ratingVector[5], num_neighbors)
print('Scores: %s' % scores)

# Function to convert   
 
    
    # initialize an empty string 
listToStr = ' '.join([str(elem) for elem in scores]) 
        
        
# Driver code     
print(listToStr)  
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

