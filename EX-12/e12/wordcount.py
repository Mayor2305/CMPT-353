import sys
from pyspark.sql import SparkSession, functions, types, Row
import re
import numpy as np
import math
import string

spark = SparkSession.builder.appName('correlate logs').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

wordbreak = r'[%s\s]+' % (re.escape(string.punctuation),)  # regex that matches spaces and/or punctuation
    
def main(in_directory, out_directory):
    data = spark.read.text(in_directory)
    split = data.select(functions.split(data.value, wordbreak).alias('words'))
    
    data = split.select(functions.explode(split.words).alias("word")).cache()
    data = data.select(functions.lower(data.word).alias('word'))
    data = data.groupby('word').count()
    data = data.sort(functions.col('count').desc(), functions.col('word'))
    # data = data.filter(data.word != " ")
    data = data.filter(data.word != "")
    data.write.csv(out_directory, mode='overwrite')

if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)         
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    