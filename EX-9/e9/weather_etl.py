import sys
from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('weather ETL').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.4' # make sure we have Spark 2.4+

observation_schema = types.StructType([
    types.StructField('station', types.StringType()),
    types.StructField('date', types.StringType()),
    types.StructField('observation', types.StringType()),
    types.StructField('value', types.IntegerType()),
    types.StructField('mflag', types.StringType()),
    types.StructField('qflag', types.StringType()),
    types.StructField('sflag', types.StringType()),
    types.StructField('obstime', types.StringType()),
])


def main(in_directory, out_directory):

    weather = spark.read.csv(in_directory, schema=observation_schema)

    # TODO: finish here.
    
    #Keep only the records we care about:
    weather = weather.filter(weather.qflag.isNull()) # --> field qflag (quality flag) is null; 
    weather = weather.filter(weather.station.startswith('CA'))  # --> the station starts with 'CA'; (option 1)
    weather = weather.filter(weather.observation == 'TMAX') # --> the observation is 'TMAX'.
    
    # Divide the temperature by 10 so it's actually in °C, and call the resulting column tmax.
    weather = weather.withColumn("tmax", weather.value / 10)
    
    # Keep only the columns station, date, and tmax
    drop = [ 'observation', 'value', 'mflag', 'qflag', 'sflag', 'obstime' ]
    cleaned_data = weather.drop(*drop)
    
    cleaned_data.write.json(out_directory, compression='gzip', mode='overwrite')


if __name__=='__main__':
    in_directory = sys.argv[1]
    out_directory = sys.argv[2]
    main(in_directory, out_directory)