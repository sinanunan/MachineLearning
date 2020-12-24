A movie recommendation algorithm that uses three different models built on top of an abstract class.

Model 1: Only predicts the mean of ratings (Used as a baseline)
Model 2: Predicts with the mean, and a scalar bias value for the user and a scalar bias value for the movie
Model 3: On top of Model 2, uses matrix factorization to extract factors for each movie and user and makes
         a prediction based on that. 
      
