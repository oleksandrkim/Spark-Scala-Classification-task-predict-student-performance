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

**Some data preprocessing**

```
val df_pass = data.withColumn("pass", when($"G3" >= 10 , 1).otherwise(0)).drop("G1").drop("G2").drop("G3")

val df_label = df_pass.select($"school", $"sex", $"age", $"address", $"famsize", $"Pstatus", $"Medu", $"Fedu", $"Mjob", $"Fjob",
  $"reason", $"guardian", $"traveltime", $"studytime", $"failures", $"schoolsup", $"famsup", $"paid", $"activities", $"nursery",
  $"higher", $"internet", $"romantic", $"famrel", $"freetime", $"goout", $"Dalc", $"Walc", $"health", $"absences", df_pass("pass").as("label"))
```
 
**Encode categorical variables: convert strings to integers and encode with OneHotEncoderEstimator**

```
// Import VectorAssembler and Vectors
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors

//categorical_var = [0,1,3,4,5,8,9,10,11,15,16,17,18,19,20,21,22]

val schoolIndexer = new StringIndexer().setInputCol("school").setOutputCol("schoolIndex")
val sexIndexer = new StringIndexer().setInputCol("sex").setOutputCol("sexIndex")
val addressIndexer = new StringIndexer().setInputCol("address").setOutputCol("addressIndex")
val famsizeIndexer = new StringIndexer().setInputCol("famsize").setOutputCol("famsizeIndex")
val PstatusIndexer = new StringIndexer().setInputCol("Pstatus").setOutputCol("PstatusIndex")
val MjobIndexer = new StringIndexer().setInputCol("Mjob").setOutputCol("MjobIndex")
val FjobIndexer = new StringIndexer().setInputCol("Fjob").setOutputCol("FjobIndex")
val reasonIndexer = new StringIndexer().setInputCol("reason").setOutputCol("reasonIndex")
val guardianIndexer = new StringIndexer().setInputCol("guardian").setOutputCol("guardianIndex")
val schoolsupIndexer = new StringIndexer().setInputCol("schoolsup").setOutputCol("schoolsupIndex")
val famsupIndexer = new StringIndexer().setInputCol("famsup").setOutputCol("famsupIndex")
val paidIndexer = new StringIndexer().setInputCol("paid").setOutputCol("paidIndex")
val activitiesIndexer = new StringIndexer().setInputCol("activities").setOutputCol("activitiesIndex")
val nurseryIndexer = new StringIndexer().setInputCol("nursery").setOutputCol("nurseryIndex")
val higherIndexer = new StringIndexer().setInputCol("higher").setOutputCol("higherIndex")
val internetIndexer = new StringIndexer().setInputCol("internet").setOutputCol("internetIndex")
val romanticIndexer = new StringIndexer().setInputCol("romantic").setOutputCol("romanticIndex")

import org.apache.spark.ml.feature.OneHotEncoderEstimator

val encoder = new OneHotEncoderEstimator().setInputCols(Array("schoolIndex", "sexIndex", "addressIndex", "famsizeIndex", "PstatusIndex", "Medu", "Fedu", "MjobIndex", "FjobIndex",
  "reasonIndex", "guardianIndex", "traveltime", "studytime", "failures", "schoolsupIndex", "famsupIndex", "paidIndex", "activitiesIndex", "nurseryIndex",
  "higherIndex", "internetIndex", "romanticIndex", "famrel", "freetime", "goout", "Dalc", "Walc", "health")).setOutputCols(Array("schoolEnc",
  "sexEnc", "addressEnc", "famsizeEnc", "PstatusEnc", "MeduEnc", "FeduEnc", "MjobEnc", "FjobEnc",
  "reasonEnc", "guardianEnc", "traveltimeEnc", "studytimeEnc", "failuresEnc", "schoolsupEnc", "famsupEnc", "paidIndexEnc", "activitiesEnc", "nurseryEnc",
  "higherEnc", "internetEnc", "romanticEnc", "famrelEnc", "freetimeEnc", "gooutEnc", "DalcEnc", "WalcEnc", "healthEnc"))

```


**Vector assembler**

```
val assembler = (new VectorAssembler()
                  .setInputCols(Array("schoolEnc", "sexEnc", "addressEnc", "famsizeEnc", "PstatusEnc", "MeduEnc", "FeduEnc", "MjobEnc", "FjobEnc",
                  "reasonEnc", "guardianEnc", "traveltimeEnc", "studytimeEnc", "failuresEnc", "schoolsupEnc", "famsupEnc", "paidIndexEnc", "activitiesEnc", "nurseryEnc",
                  "higherEnc", "internetEnc", "romanticEnc", "famrelEnc", "freetimeEnc", "gooutEnc", "DalcEnc", "WalcEnc", "healthEnc", "age", "absences"))
                  .setOutputCol("features_assem") )
```

**Scalling of features with MinMaxScaler**

```
import org.apache.spark.ml.feature.MinMaxScaler
val scaler = new MinMaxScaler().setInputCol("features_assem").setOutputCol("features")
```

**Train/Test split**

```
val Array(training, test) = df_label.randomSplit(Array(0.75, 0.25))
```

## Decision Tree

**Building a decision tree, contructing a pipeline and creating a ParamGrid**

Parameters: Max depth(5, 10, 15, 20, 30) and Max Bins(10, 20, 30, 50)

```
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

val dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("features")//.setImpurity("variance")

val pipeline = new Pipeline().setStages(Array(schoolIndexer,sexIndexer,addressIndexer,famsizeIndexer,PstatusIndexer,MjobIndexer,FjobIndexer,reasonIndexer,
  guardianIndexer,schoolsupIndexer,famsupIndexer,paidIndexer,activitiesIndexer,nurseryIndexer,higherIndexer,internetIndexer,romanticIndexer,encoder, assembler,scaler, dt))

val paramGrid = new ParamGridBuilder().addGrid(dt.maxDepth, Array(5, 10, 15, 20, 30)).addGrid(dt.maxBins, Array(10, 20, 30, 50)).build()
```

**Cross-validation (3 splits); Predict test data**

```
val cv = new CrossValidator().setEstimator(pipeline).setEvaluator(new BinaryClassificationEvaluator).setEstimatorParamMaps(paramGrid).setNumFolds(3)
val cvModel = cv.fit(training)
val predictions = cvModel.transform(test)
```
