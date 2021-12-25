import sys
from pyspark.sql import SparkSession, functions, types, Row
import re
import numpy as np
import math

spark = SparkSession.builder.appName('correlate logs').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+

line_re = re.compile(r"^(\S+) - - \[\S+ [+-]\d+\] \"[A-Z]+ \S+ HTTP/\d\.\d\" \d+ (\d+)$")


def line_to_row(line):
    """
    Take a logfile line and return a Row object with hostname and bytes transferred. Return None if regex doesn't match.
    """
    m = line_re.match(line)
    if m:
        # TODO
        return Row(host_name = m.group(1), number_of_bytes = int(m.group(2)))
    else:
        return None


def not_none(row):
    """
    Is this None? Hint: .filter() with it.
    """
    return row is not None


def create_row_rdd(in_directory):
    log_lines = spark.sparkContext.textFile(in_directory)
    # TODO: return an RDD of Row() objects
    row = log_lines.map(line_to_row)
    row = row.filter(not_none)
    return row


def main(in_directory):
    logs = spark.createDataFrame(create_row_rdd(in_directory)).cache()
    groups_1 = logs.groupby("host_name").count()
    groups_2 = logs.groupby("host_name").sum("number_of_bytes")
    data = groups_1.join(groups_2, "host_name").cache()
    data = data.withColumn('x_i_2', data['count'] * data['count'])
    data = data.withColumn('y_i_2', data['sum(number_of_bytes)'] * data['sum(number_of_bytes)'])
    data = data.withColumn('x_i_y_i', data['sum(number_of_bytes)'] * data['count'])
    n = data.count()
    data = data.groupby().sum()
    val_1 = (n * data.first()['sum(x_i_y_i)'])
    val_2 = (data.first()['sum(count)'] * data.first()['sum(sum(number_of_bytes))'])
    val_3 = math.sqrt( (n * data.first()['sum(x_i_2)'] ) - (data.first()['sum(count)'] * data.first()['sum(count)']) )
    val_4 = math.sqrt( (n * data.first()['sum(y_i_2)'] ) - (data.first()['sum(sum(number_of_bytes))'] * data.first()['sum(sum(number_of_bytes))']))
    # TODO: calculate r.
    #sum(count)|sum(sum(number_of_bytes))|sum(x_i_2)|    sum(y_i_2)|sum(x_i_y_i)
    
    r = ( val_1 - val_2)/( val_3 * val_4)# TODO: it isn't zero.
    print("r = %g\nr^2 = %g" % (r, r**2))


if __name__=='__main__':
    in_directory = sys.argv[1]
    main(in_directory)