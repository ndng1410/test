from pyspark.sql import SparkSession
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt

# Create a SparkSession
spark = SparkSession.builder \
    .appName("Attendance Prediction") \
    .getOrCreate()

# Read CSV files into Spark DataFrames
attendance_data = spark.read.csv("attendance_update.csv", header=True, inferSchema=True)
average_data = spark.read.csv('avg_grade_update.csv', header=True, inferSchema=True)
fail_count_data = spark.read.csv('fail_count_total.csv', header=True, inferSchema=True)
status_data = spark.read.csv("status_update.csv", header=True, inferSchema=True)

# Join the DataFrames
df_proc = attendance_data.join(status_data, attendance_data.User_Code == status_data.Student_Code) \
    .join(average_data, "User_Code") \
    .join(fail_count_data, "User_Code") \
    .filter(status_data.Semester == 3) \
    .drop(*['Term', 'Up_To_Semester', 'Term_ID', 'User_Login', 'Major_ID', 'Value', 'Student_Code', 'Campus_Code', 'Semester', 'Campus']) \
    .withColumn("Status_Group", when(col("Status") == "THO", 0).otherwise(1)) \
    .drop("Status") \
    .fillna(0, subset=['Average_Grade'])

print(df_proc)

# Scale features
# assembler = VectorAssembler(inputCols=df_proc.columns[:-1], outputCol="features")
# scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures", min=0, max=10)
# pipeline = Pipeline(stages=[assembler, scaler])
# pipeline_model = pipeline.fit(df_proc)
# df_scaled = pipeline_model.transform(df_proc)

# # Split the data into training and testing sets
# (training_data, test_data) = df_scaled.randomSplit([0.7, 0.3], seed=42)

# # Train the Decision Tree model
# dt = DecisionTreeClassifier(featuresCol="scaledFeatures", labelCol="Status_Group", maxDepth=5)
# dt_model = dt.fit(training_data)

# # Make predictions
# predictions = dt_model.transform(test_data)

# # Evaluate the model
# evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="Status_Group", metricName="areaUnderROC")
# auc = evaluator.evaluate(predictions)

# # Print AUC score
# print("AUC score:", auc)

# # Confusion Matrix
# conf_matrix = predictions.groupBy("Status_Group", "prediction").count()
# conf_matrix.show()

# # Plot correlation heatmap
# correlation_matrix = df_proc.toPandas().corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True)
# plt.title('Correlation Heatmap')
# plt.show()

# Close the SparkSession
spark.stop()
