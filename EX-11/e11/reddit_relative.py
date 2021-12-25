import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('reddit relative scores').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

comments_schema = types.StructType([
    types.StructField('archived', types.BooleanType()),
    types.StructField('author', types.StringType()),
    types.StructField('author_flair_css_class', types.StringType()),
    types.StructField('author_flair_text', types.StringType()),
    types.StructField('body', types.StringType()),
    types.StructField('controversiality', types.LongType()),
    types.StructField('created_utc', types.StringType()),
    types.StructField('distinguished', types.StringType()),
    types.StructField('downs', types.LongType()),
    types.StructField('edited', types.StringType()),
    types.StructField('gilded', types.LongType()),
    types.StructField('id', types.StringType()),
    types.StructField('link_id', types.StringType()),
    types.StructField('name', types.StringType()),
    types.StructField('parent_id', types.StringType()),
    types.StructField('retrieved_on', types.LongType()),
    types.StructField('score', types.LongType()),
    types.StructField('score_hidden', types.BooleanType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('subreddit_id', types.StringType()),
    types.StructField('ups', types.LongType()),
    #types.StructField('year', types.IntegerType()),
    #types.StructField('month', types.IntegerType()),
])


def main(in_directory, out_directory):
    comments = spark.read.json(in_directory, schema=comments_schema)
    
    # TODO: calculate averages
    averages = comments.groupBy("subreddit").avg("score")
    
    # Exclude any subreddits with average score ≤0.
    averages = averages.filter(averages['avg(score)'] > 0)
    
    # Join the average score to the collection of all comments. Divide to get the relative score.
    data = comments.join(averages.hint('broadcast'), "subreddit")
    data = data.withColumn("relative_score", (data["score"] / data["avg(score)"])).cache()
    
    # Determine the max relative score for each subreddit.
    groups_1 = data.groupby('subreddit')
    groups_1 = groups_1.agg(functions.max('relative_score').alias('relative_score')).cache()
    
    # Join again to get the best comment on each subreddit: we need this step to get the author.
    groups_2 = comments.groupby('subreddit')
    groups_2 = groups_2.agg(functions.max('score').alias('score')).cache()
    best_author = groups_2.join(groups_1.hint('broadcast'), ["subreddit"])
    
    # data_1.show()
    best_author = comments.join(best_author.hint('broadcast'), ['subreddit', 'score'])
    best_author = best_author['subreddit', 'author', 'relative_score']
    
    # data.show()
    best_author.write.json(out_directory, mode='overwrite')


if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)