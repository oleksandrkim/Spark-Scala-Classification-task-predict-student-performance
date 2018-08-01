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


> |-- sex: string (nullable = true) <br />
> |-- age: integer (nullable = true) <br />
> |-- address: string (nullable = true) <br />
> |-- famsize: string (nullable = true) <br />
> |-- Pstatus: string (nullable = true) <br />
> |-- Medu: integer (nullable = true) <br />
> |-- Fedu: integer (nullable = true) <br />
> |-- Mjob: string (nullable = true) <br />
> |-- Fjob: string (nullable = true) <br />
> |-- reason: string (nullable = true) <br />
> |-- guardian: string (nullable = true) <br />
> |-- traveltime: integer (nullable = true) <br />
> |-- studytime: integer (nullable = true) <br />
> |-- failures: integer (nullable = true) <br />
> |-- schoolsup: string (nullable = true) <br />
> |-- famsup: string (nullable = true) <br />
> |-- paid: string (nullable = true) <br />
> |-- activities: string (nullable = true) <br />
> |-- nursery: string (nullable = true) <br />
> |-- higher: string (nullable = true) <br />
> |-- internet: string (nullable = true) <br />
> |-- romantic: string (nullable = true) <br />
> |-- famrel: integer (nullable = true) <br />
> |-- freetime: integer (nullable = true) <br />
> |-- goout: integer (nullable = true) <br />
> |-- Dalc: integer (nullable = true) <br />
> |-- Walc: integer (nullable = true) <br />
> |-- health: integer (nullable = true) <br />
> |-- absences: integer (nullable = true) <br />
> |-- G1: integer (nullable = true) <br />
> |-- G2: integer (nullable = true) <br />
> |-- G3: integer (nullable = true) <br />



