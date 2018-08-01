# Spark-Scala-Classification-task-predict-student-performance
Predict if student passes a course or not on a basis of his or her personal life, activities in school and outside

**Information about the dataset**
- Number of inputs: 650
- Number of features: 29
- Source of data: https://archive.ics.uci.edu/ml/datasets/student+performance

**Import libraries and start of spark session**

```
// To see less warnings
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR) //less warnings pop up


// Start a simple Spark Session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()
```

***Import dataset and print schema***

```
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("student-por.csv")
data.printSchema()
```


