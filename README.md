# Machine Learning Pipeline for Movie Recommender System build by Apache Spark's ML Libary

```python
from pyspark import SparkConf
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, Normalizer
from pyspark.sql import SparkSession, functions, types
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
```


```python
spark = SparkSession.builder.master("local").appName("Recommendation").getOrCreate()
movies= spark.read.csv("./movies.csv", inferSchema = True, header = True)
ratings = spark.read.csv("./ratings.csv", inferSchema = True, header = True)
```

    22/07/08 00:17:59 WARN Utils: Your hostname, DESKTOP-LR48F7J resolves to a loopback address: 127.0.1.1; using 172.23.223.31 instead (on interface eth0)
    22/07/08 00:17:59 WARN Utils: Set SPARK_LOCAL_IP if you need to bind to another address
    22/07/08 00:18:00 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
    Using Spark's default log4j profile: org/apache/spark/log4j-defaults.properties
    Setting default log level to "WARN".
    To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).



```python
movies.show(5, truncate = False)
ratings.show(5)
```

    +-------+----------------------------------+-------------------------------------------+
    |movieId|title                             |genres                                     |
    +-------+----------------------------------+-------------------------------------------+
    |1      |Toy Story (1995)                  |Adventure|Animation|Children|Comedy|Fantasy|
    |2      |Jumanji (1995)                    |Adventure|Children|Fantasy                 |
    |3      |Grumpier Old Men (1995)           |Comedy|Romance                             |
    |4      |Waiting to Exhale (1995)          |Comedy|Drama|Romance                       |
    |5      |Father of the Bride Part II (1995)|Comedy                                     |
    +-------+----------------------------------+-------------------------------------------+
    only showing top 5 rows
    
    +------+-------+------+---------+
    |userId|movieId|rating|timestamp|
    +------+-------+------+---------+
    |     1|      1|   4.0|964982703|
    |     1|      3|   4.0|964981247|
    |     1|      6|   4.0|964982224|
    |     1|     47|   5.0|964983815|
    |     1|     50|   5.0|964982931|
    +------+-------+------+---------+
    only showing top 5 rows
    


First, for the ratings table, I use **groupBy()** to group data with the same movieId, **count()** to count the number of ratings for each movieId, and generate a new column "count", and then use **orderBy()** and **functions.desc()** to sort in descending order based on "count", and then use **limit()** to get the ten with the highest counts, so that to get ids of the top-10 movies with the largest number of ratings

Finally, use the id of the top10 movie and the movies table to perform a left join based on "movieId" column, and then use **select()** to select the “title” column to get the top10 movie names


```python
topIds = ratings.groupBy("movieId").count().orderBy(functions.desc("count")).limit(10).select("movieId")
topNames = topIds.join(movies, topIds["movieId"] == movies["movieId"], "left").select("title")
topNames.show(truncate = False)
```

    [Stage 8:===============================================>       (171 + 1) / 200]

    +-----------------------------------------+
    |title                                    |
    +-----------------------------------------+
    |Forrest Gump (1994)                      |
    |Shawshank Redemption, The (1994)         |
    |Pulp Fiction (1994)                      |
    |Silence of the Lambs, The (1991)         |
    |Matrix, The (1999)                       |
    |Star Wars: Episode IV - A New Hope (1977)|
    |Jurassic Park (1993)                     |
    |Braveheart (1995)                        |
    |Terminator 2: Judgment Day (1991)        |
    |Schindler's List (1993)                  |
    +-----------------------------------------+
    


                                                                                    

Since each movie may belong to multiple genres, I first used **functions.explode()** and **functions.split()** split the movie data belonging to multiple genres into multiple lines, each line containing exactly one genre.

Then I used **groupBy()** and **avg()** on the ratings table to average all ratings for each movie.

After this, I use **join()** to join the previous results together: i.e. each row of data should contain the movie's title, its genre, and the user's average rating for it.

Finally, for each genre's movie, use **filter()** to get all the data of movies with this genre, and use the method of the previous question to sort the data in descending order based on the average rating, and get the titles of the top 10.

For simplicity, only the results of the first three gernes are printed.


```python
moviesG = movies.withColumn("genres", functions.explode(functions.split("genres", "\\|")))
ratingsAvg = ratings.groupBy("movieId").avg("rating")
moviesGR = moviesG.join(ratingsAvg, moviesG["movieId"]==
                                 ratingsAvg["movieId"], "left").select("title", "genres", "avg(rating)")

count = 0
for genre in moviesGR.select("genres").distinct().collect():
    print(genre[0])
    moviesGR.filter(moviesGR["genres"] == genre[0]).orderBy(
        functions.desc("avg(rating)")).limit(10).select("title").show(truncate=False)
    count += 1
    if (count == 3): break
```

                                                                                    

    Crime
    +-------------------------------------------------------+
    |title                                                  |
    +-------------------------------------------------------+
    |Ex Drummer (2007)                                      |
    |Villain (1971)                                         |
    |Mother (Madeo) (2009)                                  |
    |Going Places (Valseuses, Les) (1974)                   |
    |12 Angry Men (1997)                                    |
    |American Friend, The (Amerikanische Freund, Der) (1977)|
    |Sisters (Syostry) (2001)                               |
    |Little Murders (1971)                                  |
    |Faster (2010)                                          |
    |Decalogue, The (Dekalog) (1989)                        |
    +-------------------------------------------------------+
    
    Romance


                                                                                    

    +----------------------------------------------------------------+
    |title                                                           |
    +----------------------------------------------------------------+
    |All the Vermeers in New York (1990)                             |
    |Cruel Romance, A (Zhestokij Romans) (1984)                      |
    |Bossa Nova (2000)                                               |
    |Sandpiper, The (1965)                                           |
    |Duel in the Sun (1946)                                          |
    |Moscow Does Not Believe in Tears (Moskva slezam ne verit) (1979)|
    |Continental Divide (1981)                                       |
    |Seems Like Old Times (1980)                                     |
    |Man and a Woman, A (Un homme et une femme) (1966)               |
    |Rain (2001)                                                     |
    +----------------------------------------------------------------+
    
    Thriller
    +-------------------------------------------------------+
    |title                                                  |
    +-------------------------------------------------------+
    |Cherish (2002)                                         |
    |Maniac Cop 2 (1990)                                    |
    |American Friend, The (Amerikanische Freund, Der) (1977)|
    |What Happened Was... (1994)                            |
    |I, the Jury (1982)                                     |
    |'Salem's Lot (2004)                                    |
    |Mother (Madeo) (2009)                                  |
    |Breed, The (2006)                                      |
    |Assignment, The (1997)                                 |
    |Supercop 2 (Project S) (Chao ji ji hua) (1993)         |
    +-------------------------------------------------------+
    


I first select the "movieId", "userId" columns in the ratings table and the ids of the first 100 movies in the movies table.

Use two for loops to iterate through all movie pairs, the second loop starts from the position already traversed in the first loop to avoid double counting, use **filter()** to find the ratings for the first item and second item in movie pairs respectively, use inner join to find users who rated both items, and use **count()** to calculate the number of users who rated both items.

For simplicity, only the first 5 results of are printed.


```python
movieUserId = ratings.select("movieId", "userId")
frist100MovieId = movies.select("movieId").limit(100).collect()

count = 0
for i, movieId1 in enumerate(frist100MovieId):
    for movieId2 in frist100MovieId[i+1:]:
        userMovieId1 = movieUserId.filter(movieUserId["movieId"]==movieId1[0])
        userMovieId2 = movieUserId.filter(movieUserId["movieId"]==movieId2[0])
        commonSupport = userMovieId1.join(userMovieId2, userMovieId1["userId"]
                                          ==userMovieId2["userId"], "inner").count()
        print("(" + str(movieId1[0]) + "," + str(movieId2[0]) + ") : " + str(commonSupport))
        count += 1
        if (count == 5): break
    break
```

    (1,2) : 68
    (1,3) : 32
    (1,4) : 2
    (1,5) : 32
    (1,6) : 58


I used **randomSplit()** to split the data into training set and test set.
Using explicit feedback from **ASL()** to build a recommendation model.
Use the training set to train the model and use the test set to predict user ratings for movies


```python
trainingSet, testSet = ratings.randomSplit([8., 2.])
alsExplicit = ALS(maxIter=5, regParam=0.01, userCol="userId", 
                  itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
modelExplicit = alsExplicit.fit(trainingSet)
predictionsExplicit = modelExplicit.transform(testSet)
predictionsExplicit.show()
```

    22/07/08 00:18:15 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
    22/07/08 00:18:15 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
    22/07/08 00:18:15 WARN LAPACK: Failed to load implementation from: com.github.fommil.netlib.NativeSystemLAPACK
    22/07/08 00:18:15 WARN LAPACK: Failed to load implementation from: com.github.fommil.netlib.NativeRefLAPACK
    [Stage 89:================================================>     (181 + 1) / 200]

    +------+-------+------+----------+----------+
    |userId|movieId|rating| timestamp|prediction|
    +------+-------+------+----------+----------+
    |   597|    471|   2.0| 941558175| 3.9375226|
    |   385|    471|   4.0| 850766697|  3.910093|
    |   602|    471|   4.0| 840876085| 2.3489313|
    |    91|    471|   1.0|1112713817| 3.6434646|
    |   136|    471|   4.0| 832450058|  6.453267|
    |   273|    471|   5.0| 835861348|  5.202881|
    |   448|    471|   4.0|1178980875| 2.7267814|
    |   541|    471|   3.0| 835643551| 4.0516706|
    |   104|    471|   4.5|1238111129| 3.6285505|
    |   307|    833|   1.0|1186172725| 1.2996885|
    |   177|   1088|   3.5|1435534616| 3.3767974|
    |   474|   1088|   3.5|1100292226|  2.784757|
    |    20|   1088|   4.5|1054147512| 3.3778894|
    |   479|   1088|   4.0|1039362157| 3.1606584|
    |   489|   1088|   4.5|1332775009| 2.6833987|
    |   381|   1088|   3.5|1168664508| 3.7349133|
    |    84|   1088|   3.0| 860398568| 3.7105742|
    |    51|   1088|   4.0|1230929736| 2.6063488|
    |   414|   1088|   3.0| 961514273| 3.4906068|
    |   587|   1238|   4.0| 953138576|  3.106637|
    +------+-------+------+----------+----------+
    only showing top 20 rows
    


                                                                                    

Evaluate the model based on the root mean squared error of rating predictions


```python
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol ="prediction")
rmseExplicit = evaluator.evaluate(predictionsExplicit)
print("Explicit:Root-mean-square error = "+str(rmseExplicit))
```

    [Stage 108:===============================================>     (181 + 1) / 200]

    Explicit:Root-mean-square error = 1.068086127938262


                                                                                    

For a user(Id), **getRecommend()** returns the top 5 movies with the highest predictions of ratings, thus completing the recommendation.


```python
def getRecommend(userId):
    recommendIds = predictionsExplicit.filter(predictionsExplicit["userId"]
                                              ==userId).orderBy(functions.desc("prediction")).limit(5)
    
    recommendMovies = recommendIds.join(movies, recommendIds["movieId"]
                                        ==movies["movieId"], "left").select("title")
    
    recommendMovies.show(truncate=False)
    
getRecommend(5)
```

    [Stage 128:===================================================> (194 + 1) / 200]

    +--------------------------------------+
    |title                                 |
    +--------------------------------------+
    |Postman, The (Postino, Il) (1994)     |
    |Fargo (1996)                          |
    |Quiz Show (1994)                      |
    |Pulp Fiction (1994)                   |
    |Snow White and the Seven Dwarfs (1937)|
    +--------------------------------------+
    


                                                                                    
