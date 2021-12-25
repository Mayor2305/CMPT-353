import sys
import re
from datetime import datetime
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('reddit averages').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

schema = types.StructType([
    types.StructField('language', types.StringType()),
    types.StructField('title', types.StringType()),
    types.StructField('views', types.LongType()),
    types.StructField('bits', types.StringType()),
])

def datetime_function(data):
    datetime = re.search("([0-9]{8}\-[0-9]{2})", data)
    return datetime.group(1)

def main(in_directory, out_directory):
    page_data = spark.read.csv(in_directory, sep = " ", schema=schema).withColumn('filename', functions.input_file_name())
    page_data = page_data.filter(page_data["language"] == "en")
    page_data = page_data.filter(page_data["title"] != "Main_Page")
    page_data = page_data.filter(page_data["title"].startswith("Special:") == 0)
    
    path_to_hour = functions.udf(datetime_function, returnType=types.StringType())
    page_data = page_data.withColumn("datetime", path_to_hour(page_data.filename))
    page_data = page_data ['datetime', 'title', 'views']
    
    groups = page_data.groupby('datetime')
    page_data = page_data.cache()
    
    data = groups.max('views')
    data = data.join(page_data, on='datetime')
    data = data.filter(data['views']==data['max(views)'])
    data = data.sort('datetime','title')
    data = data['datetime', 'title', 'max(views)']
    data.write.format('csv').mode('overwrite').save(out_directory)
    
if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)