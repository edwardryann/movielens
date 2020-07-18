# Downloading Data --------------------------------------------------------

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate")
if(!require(recosystem)) install.packages("recosystem")
if(!require(gridExtra)) install.packages("gridExtra")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# RMSE function -----------------------------------------------------------
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Saving workspace image --------------------------------------------------
save.image(file=".Rdata")

# Reset workspace ---------------------------------------------------------
rm(list = setdiff(ls(), c("edx", "validation", "RMSE")))
# Reset edx and validation columns
edx <- edx[1:6]
validation <- validation[1:6]
# Libraries ---------------------------------------------------------------
library(tidyverse)
library(lubridate)
library(caret)
library(recosystem)
library(data.table)
library(gridExtra)
# RMSE Model Results ------------------------------------------------------------
rmse_results <- tibble()
# Data Exploration --------------------------------------------------------

# Overview of the dataset
head(edx)
edx %>%
  ggplot(aes(rating)) +
  geom_density(adjust = 30) + 
  geom_vline(xintercept = mean(edx$rating), linetype = "dotted", color = "red") +
  ggtitle("Ratings distribution")

# Sparse data
temp <- edx %>%
  filter(userId %in% sample(edx$userId, 100, replace = FALSE)) %>%
  select(userId, movieId, rating) %>%
  spread(key = movieId, value = rating)
ncol <- ncol(temp)
temp <- temp %>% select(sample(1:ncol, 100, replace = FALSE)) %>% as.matrix()
temp[!is.na(temp)] <- 1
temp %>% image(xlab = "Movies", ylab = "Users", axes = FALSE, main = "Sparse Data")
axis(1, at = seq(0.0,1.0,0.2), labels = seq(0,100,20))
axis(2, at = seq(0.0,1.0,0.2), labels = seq(0,100,20))
rm(temp)

# Different movie and user averages
plot_1 <- edx %>%
  group_by(movieId) %>%
  summarize(avg = mean(rating)) %>%
  ggplot(aes(avg)) +
  geom_histogram() +
  ggtitle("Movie averages")

plot_2 <- edx %>%
  group_by(userId) %>%
  summarize(avg = mean(rating)) %>%
  ggplot(aes(avg)) +
  geom_histogram() +
  ggtitle("User averages")

grid.arrange(plot_1, plot_2)
rm(plot_1, plot_2)

# Different timestamp averages

# Mutate edx
edx <- edx %>%
  mutate(day = timestamp %>% as_datetime() %>% round_date(unit = "day"),
         year = timestamp %>% as_datetime() %>% round_date(unit = "year"))

# Year averages
plot_1 <- edx %>%
  group_by(year) %>%
  summarize(avg = mean(rating)) %>%
  ggplot(aes(year, avg)) + 
  geom_point() +
  geom_smooth() +
  ggtitle("Year averages")

# Day averages
plot_2 <- edx %>%
  group_by(day) %>%
  summarize(avg = mean(rating)) %>%
  ggplot(aes(day, avg)) + 
  geom_point(alpha = 0.2) +
  geom_smooth() +
  ggtitle("Day averages")

grid.arrange(plot_1, plot_2, ncol = 2)

# Release date averages
# Extracting release date
release_date <- edx$title %>%
  str_extract(pattern = "\\(\\d{4}\\)$") %>%
  str_replace(pattern = "\\(", replacement = "") %>%
  str_replace(pattern = "\\)", replacement = "")
edx <- edx %>%
  mutate(release_date = release_date)
rm(release_date)

edx %>%
  mutate(release_date = as_date(release_date, format = "%Y")) %>%
  group_by(release_date) %>%
  summarize(avg = mean(rating)) %>%
  ggplot(aes(release_date, avg)) +
  geom_point() + 
  geom_smooth() +
  ggtitle("Release date averages")

# Age upon review averages

# Adding age by year
age <- as.Date(edx$year) - as.Date(edx$release_date, format = "%Y")
edx <- edx %>%
  mutate(age_year = age)
rm(age)

edx %>%
  group_by(age_year) %>%
  summarize(avg = mean(rating)) %>%
  ggplot(aes(age_year, avg)) + 
  geom_point() +
  geom_smooth() +
  ggtitle("Age upon review averages")

# Genre averages
edx %>%
  group_by(genres) %>%
  summarize(avg = mean(rating)) %>%
  arrange(avg) %>%
  ggplot(aes(x = reorder(genres, avg), y = avg)) +
  geom_col() +
  theme(axis.text.x = element_blank()) +
  xlab("genres") +
  ggtitle("Genre averages")

# Regularization needed
# n Ratings for Movies
plot_1 <- edx %>%
  group_by(movieId) %>%
  summarize(n = n()) %>%
  filter(n < 10) %>%
  ggplot(aes(n)) +
  geom_bar() +
  ggtitle("n Ratings for Movies") +
  scale_x_continuous(breaks = seq(1:9))

# n Ratings for Users
plot_2 <- edx %>%
  group_by(userId) %>%
  summarize(n = n()) %>%
  filter(n < 50) %>%
  ggplot(aes(n)) +
  geom_bar() +
  ggtitle("n Ratings for Users") +
  scale_x_continuous()

# n Ratings for Timestamp (Day)
plot_3 <- edx %>%
  group_by(day) %>%
  summarize(n = n()) %>%
  filter(n < 100) %>%
  ggplot(aes(n)) +
  geom_bar() +
  ggtitle("n Ratings for Timestamp") +
  scale_x_continuous()

# n Ratings for Genres
plot_4 <- edx %>%
  group_by(genres) %>%
  summarize(n = n()) %>%
  filter(n < 100) %>%
  ggplot(aes(n)) +
  geom_bar() +
  ggtitle("n Ratings for Genres") +
  scale_x_continuous()

# Combining plots
grid.arrange(plot_1, plot_2, plot_3, plot_4, ncol = 2, nrow = 2)
rm(plot_1, plot_2, plot_3, plot_4)

# n Ratings for Release Date
edx %>%
  group_by(release_date) %>%
  summarize(n = n()) %>%
  arrange(n) %>%
  head()

# n Ratings for Age (Year)
edx %>%
  group_by(age_year) %>%
  summarize(n = n()) %>%
  arrange(n) %>%
  head(n = 15)

#####
# Model 1 : Just the average -------------------------------------------------------
# Finding the average
mu <- mean(edx$rating)
# Getting the prediction
prediction <- rep(mu, nrow(validation))
# Finding RMSE of the model
rmse_results <- bind_rows(rmse_results,
                          tibble(Model = "Just the average",
                                 RMSE = RMSE(validation$rating, prediction)))
# Model 2 : Movie effect ------------------------------------------------------------
# Finding movie averages from residuals
movie_avg <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating-mu))

# Getting prediction
prediction <- validation %>%
  left_join(movie_avg, by = "movieId") %>%
  mutate(prediction = mu + b_i) %>%
  pull(prediction)

# Finding RMSE of the model
rmse_results <- bind_rows(rmse_results,
                          tibble(Model = "Movie Effect",
                                 RMSE = RMSE(validation$rating, prediction)))
# Model 3 : Movie + User Effects -----------------------------------------------------
# Finding user averages from residuals
user_avg <- edx %>%
  left_join(movie_avg, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Getting prediction
prediction <- validation %>%
  left_join(movie_avg, by = "movieId") %>%
  left_join(user_avg, by = "userId") %>%
  mutate(prediction = mu + b_i + b_u) %>%
  pull(prediction)

# Finding RMSE of the model
rmse_results <- bind_rows(rmse_results,
                          tibble(Model = "Movie + User Effects",
                                 RMSE = RMSE(validation$rating, prediction)))

# Model 4 : Movie + User + Timestamp Effects ------------------------------

# Mutate edx and validation, adding day and year of the ratings
edx <- edx %>%
  mutate(day = timestamp %>% as_datetime() %>% round_date(unit = "day"),
         year = timestamp %>% as_datetime() %>% round_date(unit = "year"))
validation <- validation %>%
  mutate(day = timestamp %>% as_datetime() %>% round_date(unit = "day"),
         year = timestamp %>% as_datetime() %>% round_date(unit = "year"))

# Timestamp effect, by day

# Finding day averages from residuals
day_avg <- edx %>%
  left_join(movie_avg, by = "movieId") %>%
  left_join(user_avg, by = "userId") %>%
  group_by(day) %>%
  summarize(b_t = mean(rating - mu - b_i - b_u))

# Getting prediction
prediction <- validation %>%
  left_join(movie_avg, by = "movieId") %>%
  left_join(user_avg, by = "userId") %>%
  left_join(day_avg, by = "day") %>%
  mutate(prediction = mu + + b_i + b_u + b_t) %>%
  pull(prediction)

# Finding RMSE of the model
rmse_results <- bind_rows(rmse_results,
                          tibble(Model = "Movie + User + Timestamp (Day) Effects",
                                 RMSE = RMSE(validation$rating, prediction)))

# Model 5 : Movie + User + Timestamp + Release Date Effects ---------------

# Release date effect

# Extracting release date from the title of edx
release_date <- edx$title %>%
  str_extract(pattern = "\\(\\d{4}\\)$") %>%
  str_replace(pattern = "\\(", replacement = "") %>%
  str_replace(pattern = "\\)", replacement = "")

# Mutate release date into edx
edx <- edx %>%
  mutate(release_date = release_date)

# Extracting release date from the title of validation
release_date <- validation$title %>%
  str_extract(pattern = "\\(\\d{4}\\)$") %>%
  str_replace(pattern = "\\(", replacement = "") %>%
  str_replace(pattern = "\\)", replacement = "")

# Mutate release date into validation
validation <- validation %>%
  mutate(release_date = release_date)

# Removing unused variable
rm(release_date)

# Finding release date averages from residuals
release_avg <- edx %>%
  left_join(movie_avg, by = "movieId") %>%
  left_join(user_avg, by = "userId") %>%
  left_join(day_avg, by = "day") %>%
  group_by(release_date) %>%
  summarize(b_rel = mean(rating-mu-b_i-b_u-b_t))

# Getting prediction
prediction <- validation %>%
  left_join(movie_avg, by = "movieId") %>%
  left_join(user_avg, by = "userId") %>%
  left_join(day_avg, by = "day") %>%
  left_join(release_avg, by = "release_date") %>%
  mutate(prediction = mu + b_i + b_u + b_t + b_rel) %>%
  pull(prediction)

# Finding RMSE of the model
rmse_results <- bind_rows(rmse_results,
                          tibble(Model = "Movie + User + Timestamp (Day) + Release Date Effects",
                                 RMSE = RMSE(validation$rating, prediction)))

# Model 6 : Movie + User + Timestamp + Release Date + Age Effects ---------

# Age effect

# Finding age by subtracting rating date with release date (edx)
age <- as.Date(edx$year) - as.Date(edx$release_date, format = "%Y")

# Mutate age into edx
edx <- edx %>%
  mutate(age_year = age)

# Finding age by substracting rating date with release date (validation)
age <- as.Date(validation$year) - as.Date(validation$release_date, format = "%Y")

# Mutate age into validation
validation <- validation %>%
  mutate(age_year = age)

# Removing unsued variable
rm(age)

# Finding age averages from residuals
age_avg <- edx %>%
  left_join(movie_avg, by = "movieId") %>%
  left_join(user_avg, by = "userId") %>%
  left_join(day_avg, by = "day") %>%
  left_join(release_avg, by = "release_date") %>%
  group_by(age_year) %>%
  summarize(b_age = mean(rating - mu - b_i - b_u - b_t - b_rel))

# Getting prediction
prediction <- validation %>%
  left_join(movie_avg, by = "movieId") %>%
  left_join(user_avg, by = "userId") %>%
  left_join(day_avg, by = "day") %>%
  left_join(release_avg, by = "release_date") %>%
  left_join(age_avg, by = "age_year") %>%
  mutate(prediction = mu + b_i + b_u + b_t + b_rel + b_age) %>%
  pull(prediction)

# Finding RMSE of the model
rmse_results <- bind_rows(rmse_results,
                          tibble(Model = "Movie + User + Timestamp (Day) + Release Date + Age (Year) Effects",
                                 RMSE = RMSE(validation$rating, prediction)))


# Model 7 : Movie + User + Timestamp + Release Date + Age + Genre Effects -----------------------------------------------------------

# Finding genre averages from residuals
genre_avg <- edx %>%
  left_join(movie_avg, by = "movieId") %>%
  left_join(user_avg, by = "userId") %>%
  left_join(day_avg, by = "day") %>%
  left_join(release_avg, by = "release_date") %>%
  left_join(age_avg, by = "age_year") %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u - b_t - b_rel - b_age))

# Getting prediction
prediction <- validation %>%
  left_join(movie_avg, by = "movieId") %>%
  left_join(user_avg, by = "userId") %>%
  left_join(day_avg, by = "day") %>%
  left_join(release_avg, by = "release_date") %>%
  left_join(age_avg, by = "age_year") %>%
  left_join(genre_avg, by = "genres") %>%
  mutate(prediction = mu + b_i + b_u + b_t + b_rel + b_age + b_g) %>%
  pull(prediction)

# Finding RMSE of the model
rmse_results <- bind_rows(rmse_results,
                          tibble(Model = "Movie + User + Timestamp (Day) + Release Date + Age (Year) + Genre Effects",
                                 RMSE = RMSE(validation$rating, prediction)))


# Model 8 : Regularization Movie + User + Timestamp + Release Date + Age + Genres Effects -----------------------------------------------------------------

# Setting parameters
k <- 25
lambda <- seq(4.75,6.25,0.25)

# Creating train and test set
set.seed(1)
test_index <- createDataPartition(edx$rating, times = 1, p = 0.1, list = FALSE)
temp <- edx %>% slice(test_index)
train_set <- edx %>% slice(-test_index)
# Make sure movieId, userId, day, release date, age, and genres in test set are also in train set
test_set <- temp %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId") %>%
  semi_join(train_set, by = "day") %>%
  semi_join(train_set, by = "release_date") %>%
  semi_join(train_set, by = "age_year") %>%
  semi_join(train_set, by = "genres")
# Add rows removed from train set back into edx set
removed <- anti_join(temp, test_set) # Removed 17 rows only
train_set <- bind_rows(train_set, removed)
# Removing unused variables
rm(test_index, removed, temp)

# Create train_set partitions
set.seed(1)
folds_index <- createFolds(train_set$rating, k = k)

# Cross validation (Folds) (Finding the best lambda for regularization)
train_folds_index <- seq(1:nrow(train_set))

best_lambda <- function(lambda){
  # Setting cross validation from partitions
  sapply(1:k, function(x){
    index <- train_folds_index
    index <- index[!index %in% folds_index[[x]]]
    train_folds_set <- train_set[index,]
    temp <- train_set[(folds_index[[x]]),]
    # Make sure movieId, userId, timestamp (day), release date, age, and genres are in test set are also in train set
    test_folds_set <- temp %>%
      semi_join(train_folds_set, by = "movieId") %>%
      semi_join(train_folds_set, by = "userId") %>%
      semi_join(train_folds_set, by = "day") %>%
      semi_join(train_folds_set, by = "release_date") %>%
      semi_join(train_folds_set, by = "age_year") %>%
      semi_join(train_folds_set, by = "genres")
    # Add rows removed from test set back into train set
    removed <- anti_join(temp, test_folds_set)
    train_folds_set <- bind_rows(train_folds_set, removed)
    # Removing unused variables
    rm(index, temp, removed)
    
    # Finding averages from various effects
    movie_avg <- train_folds_set %>%
      group_by(movieId) %>%
      summarize(b_i = sum(rating-mu)/(n() + lambda))
    temp <- train_folds_set %>%
      left_join(movie_avg, by = "movieId")
    user_avg <- temp %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating-mu-b_i)/(n() + lambda))
    temp <- temp %>%
      left_join(user_avg, by = "userId")
    time_avg <- temp %>%
      group_by(day) %>%
      summarize(b_t = sum(rating - mu - b_i - b_u)/(n() + lambda))
    temp <- temp %>%
      left_join(time_avg, by = "day")
    release_avg <- temp %>%
      group_by(release_date) %>%
      summarize(b_rel = sum(rating - mu - b_i - b_u - b_t)/(n() + lambda))
    temp <- temp %>%
      left_join(release_avg, by = "release_date")
    age_avg <- temp %>%
      group_by(age_year) %>%
      summarize(b_age = sum(rating - mu - b_i - b_u - b_t - b_rel)/(n() + lambda))
    temp <- temp %>%
      left_join(age_avg, by = "age_year")
    genre_avg <- temp %>%
      group_by(genres) %>%
      summarize(b_g = sum(rating- mu - b_i - b_u - b_t - b_rel - b_age)/(n() + lambda))
    rm(temp)
    
    # Getting prediction
    prediction <- test_folds_set %>%
      left_join(movie_avg, by = "movieId") %>%
      left_join(user_avg, by = "userId") %>%
      left_join(time_avg, by = "day") %>%
      left_join(release_avg, by = "release_date") %>%
      left_join(age_avg, by = "age_year") %>%
      left_join(genre_avg, by = "genres") %>%
      mutate(prediction = mu + b_i + b_u + b_t + b_rel + b_age + b_g) %>%
      pull(prediction)
    
    # Returning results
    return(RMSE(test_folds_set$rating, prediction))
    
  })
}
# Storing results
result <- sapply(lambda, best_lambda)

# Removing unsued variables
rm(folds_index, train_folds_index)

# PLot RMSE against lambda
ggplot(mapping = aes(lambda, colMeans(result))) +
  geom_point() +
  ggtitle("Lambda vs RMSE") +
  ylab("RMSE")

# Finding best lambda
index <- colMeans(result) %>% which.min()
lambda <- lambda[index]
lambda
rm(index)


# Estimating RMSE with the test set

# Setting parameters
k <- 10

# Creating partitions
folds_index <- createFolds(edx$rating, k = k)

# Storing result
result <- sapply(1:k, function(x){
  # Setting cross validation from partition
  temp <- edx %>% slice(folds_index[[x]])
  train_set <- edx %>% slice(-folds_index[[x]])
  # Make sure movieId, userId, day, release date, age, and genres in the test set are also in the train set
  test_set <- temp %>%
    semi_join(train_set, by = "movieId") %>%
    semi_join(train_set, by = "userId") %>%
    semi_join(train_set, by = "day") %>%
    semi_join(train_set, by = "release_date") %>%
    semi_join(train_set, by = "age_year") %>%
    semi_join(train_set, by = "genres")
  # Add rows removed from test set back into train set
  removed <- anti_join(temp, test_set)
  train_set <- bind_rows(train_set, removed)
  rm(temp, removed)
  
  # Finding averages from various effects
  movie_avg <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating-mu)/(n() + lambda))
  temp <- train_set %>%
    left_join(movie_avg, by = "movieId")
  user_avg <- temp %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating-mu-b_i)/(n() + lambda))
  temp <- temp %>%
    left_join(user_avg, by = "userId")
  time_avg <- temp %>%
    group_by(day) %>%
    summarize(b_t = sum(rating - mu - b_i - b_u)/(n() + lambda))
  temp <- temp %>%
    left_join(time_avg, by = "day")
  release_avg <- temp %>%
    group_by(release_date) %>%
    summarize(b_rel = sum(rating - mu - b_i - b_u - b_t)/(n() + lambda))
  temp <- temp %>%
    left_join(release_avg, by = "release_date")
  age_avg <- temp %>%
    group_by(age_year) %>%
    summarize(b_age = sum(rating - mu - b_i - b_u - b_t - b_rel)/(n() + lambda))
  temp <- temp %>%
    left_join(age_avg, by = "age_year")
  genre_avg <- temp %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating- mu - b_i - b_u - b_t - b_rel - b_age)/(n() + lambda))
  rm(temp)
  
  # Getting prediction
  prediction <- test_set %>%
    left_join(movie_avg, by = "movieId") %>%
    left_join(user_avg, by = "userId") %>%
    left_join(time_avg, by = "day") %>%
    left_join(release_avg, by = "release_date") %>%
    left_join(age_avg, by = "age_year") %>%
    left_join(genre_avg, by = "genres") %>%
    mutate(prediction = mu + b_i + b_u + b_t + b_rel + b_age + b_g) %>%
    pull(prediction)
  
  # Returning results
  return(RMSE(test_set$rating, prediction))
})

# Removing unsued variable
rm(folds_index)

# Storing estimate of RMSE
cv_estimate <- tibble()
cv_estimate <- bind_rows(cv_estimate,
                         tibble(Model = "(Model 8) CV Estimate",
                                RMSE = mean(result)))

# Final prediction (using validation)

# Finding averages from various effects
movie_avg <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating-mu)/(n() + lambda))
temp <- edx %>%
  left_join(movie_avg, by = "movieId")
user_avg <- temp %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating-mu-b_i)/(n() + lambda))
temp <- temp %>%
  left_join(user_avg, by = "userId")
time_avg <- temp %>%
  group_by(day) %>%
  summarize(b_t = sum(rating - mu - b_i - b_u)/(n() + lambda))
temp <- temp %>%
  left_join(time_avg, by = "day")
release_avg <- temp %>%
  group_by(release_date) %>%
  summarize(b_rel = sum(rating - mu - b_i - b_u - b_t)/(n() + lambda))
temp <- temp %>%
  left_join(release_avg, by = "release_date")
age_avg <- temp %>%
  group_by(age_year) %>%
  summarize(b_age = sum(rating - mu - b_i - b_u - b_t - b_rel)/(n() + lambda))
temp <- temp %>%
  left_join(age_avg, by = "age_year")
genre_avg <- temp %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating- mu - b_i - b_u - b_t - b_rel - b_age)/(n() + lambda))
rm(temp)

# Getting prediction
prediction <- validation %>%
  left_join(movie_avg, by = "movieId") %>%
  left_join(user_avg, by = "userId") %>%
  left_join(time_avg, by = "day") %>%
  left_join(release_avg, by = "release_date") %>%
  left_join(age_avg, by = "age_year") %>%
  left_join(genre_avg, by = "genres") %>%
  mutate(prediction = mu + b_i + b_u + b_t + b_rel + b_age + b_g) %>%
  pull(prediction)

# Finding difference of the true RMSE with estimate RMSE
cv_estimate$RMSE - (RMSE(validation$rating, prediction))

# Finding RMSE of the model
rmse_results <- bind_rows(rmse_results,
                          tibble(Model = "Regularization Movie + User + Timestamp (Day) + Release Date + Age (Year) + Genres Effects ",
                                 RMSE = RMSE(validation$rating, prediction)))
# IDEA : Matrix Factorization ---------------------------------------------

# Loading library essential to perform matrix factorizaiton
library(recosystem)

lambda <- 5.25

# Finding averages from various effects (regularization)
movie_avg <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating-mu)/(n() + lambda))
temp <- edx %>%
  left_join(movie_avg, by = "movieId")
user_avg <- temp %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating-mu-b_i)/(n() + lambda))
temp <- temp %>%
  left_join(user_avg, by = "userId")
time_avg <- temp %>%
  group_by(day) %>%
  summarize(b_t = sum(rating - mu - b_i - b_u)/(n() + lambda))
temp <- temp %>%
  left_join(time_avg, by = "day")
release_avg <- temp %>%
  group_by(release_date) %>%
  summarize(b_rel = sum(rating - mu - b_i - b_u - b_t)/(n() + lambda))
temp <- temp %>%
  left_join(release_avg, by = "release_date")
age_avg <- temp %>%
  group_by(age_year) %>%
  summarize(b_age = sum(rating - mu - b_i - b_u - b_t - b_rel)/(n() + lambda))
temp <- temp %>%
  left_join(age_avg, by = "age_year")
genre_avg <- temp %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating- mu - b_i - b_u - b_t - b_rel - b_age)/(n() + lambda))
rm(temp)

# Getting residual
residual <- edx %>%
  left_join(movie_avg, by = "movieId") %>%
  left_join(user_avg, by = "userId") %>%
  left_join(time_avg, by = "day") %>%
  left_join(release_avg, by = "release_date") %>%
  left_join(age_avg, by = "age_year") %>%
  left_join(genre_avg, by = "genres") %>%
  mutate(residual = rating - mu - b_i - b_u - b_t - b_rel - b_age - b_g) %>%
  select(userId, movieId, residual)

# Matrix Factorization (Recosystem package)

# Preparing the data
edx_mf <- as.matrix(residual)
validation_mf <- validation %>% select(userId, movieId, rating) %>% as.matrix()

# Storing data in hard drive 
write.table(edx_mf, file = "train.txt", sep = " ", row.names = FALSE, col.names = FALSE)
write.table(validation_mf, file = "test.txt", sep = " ", row.names = FALSE, col.names = FALSE)
set.seed(1)

# Reading data from hard drive
train_data <- data_file("train.txt")
validation_data <- data_file("test.txt")

# Removing unused variables
rm(edx_mf, validation_mf)

# Creating model object 
r <- Reco()
# Tuning parameters
opts <- r$tune(train_data, opts = list(dim = c(10, 20, 30), lrate = c(0.1, 0.2),
                                       costp_l1 = 0, costq_l1 = 0,
                                       nthread = 1, niter = 10))

# Train data with best parameter
r$train(train_data, opts = c(opts$min, nthread = 1, niter = 20))

# Storing prediction
stored_prediction <- tempfile()
r$predict(validation_data, out_file(stored_prediction))
pred_ratings <- scan(stored_prediction)

# Making final prediction
temp <- validation %>%
  left_join(movie_avg, by = "movieId") %>%
  left_join(user_avg, by = "userId") %>%
  left_join(time_avg, by = "day") %>%
  left_join(release_avg, by = "release_date") %>%
  left_join(age_avg, by = "age_year") %>%
  left_join(genre_avg, by = "genres") %>%
  mutate(prediction = mu + b_i + b_u + b_t + b_rel + b_age + b_g) %>%
  pull(prediction)
prediction <- pred_ratings + temp
rm(temp)
rmse_results <- bind_rows(rmse_results,
                          tibble(Model = "Matrix Factorization with Best Baseline Model",
                                 RMSE = RMSE(validation$rating, prediction)))
#Removing unused variables
rm(train_data, validation_data, r, opts, stored_prediction)
