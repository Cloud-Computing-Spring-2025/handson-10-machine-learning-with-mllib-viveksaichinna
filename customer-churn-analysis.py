from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, ChiSqSelector
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Initialize Spark session
spark = SparkSession.builder.appName("CustomerChurnMLlib").getOrCreate()

# Load dataset
df = spark.read.csv("customer_churn.csv", header=True, inferSchema=True)
df = df.toDF(*[c.strip() for c in df.columns])  # Clean column names

# --- Task 1: Data Preprocessing and Feature Engineering ---
def preprocess_data(df):
    df = df.withColumn("TotalCharges", when(col("TotalCharges").isNull(), 0).otherwise(col("TotalCharges")))

    categorical_cols = ["gender", "SeniorCitizen", "PhoneService", "InternetService"]
    indexers = [StringIndexer(inputCol=c, outputCol=c + "_Index", handleInvalid="keep") for c in categorical_cols]
    encoders = [OneHotEncoder(inputCol=c + "_Index", outputCol=c + "_Vec") for c in categorical_cols]

    label_indexer = StringIndexer(inputCol="Churn", outputCol="label")
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    assembler_inputs = [c + "_Vec" for c in categorical_cols] + numeric_cols
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

    pipeline = Pipeline(stages=indexers + encoders + [label_indexer, assembler])
    model = pipeline.fit(df)
    final_df = model.transform(df).select("features", "label")

    return final_df

# --- Task 2: Train and Evaluate a Logistic Regression Model ---
def train_logistic_regression_model(df):
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    lr = LogisticRegression(featuresCol="features", labelCol="label")
    lr_model = lr.fit(train_df)
    predictions = lr_model.transform(test_df)

    evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
    print(f"✅ Logistic Regression AUC: {auc:.4f}")

# --- Task 3: Feature Selection using Chi-Square Test ---
def feature_selection(df):
    selector = ChiSqSelector(numTopFeatures=5, featuresCol="features", outputCol="selectedFeatures", labelCol="label")
    result = selector.fit(df).transform(df)
    result.select("selectedFeatures", "label").show(truncate=False)

# --- Task 4: Hyperparameter Tuning and Model Comparison ---
def tune_and_compare_models(df):
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

    models = {
        "Logistic Regression": LogisticRegression(featuresCol="features", labelCol="label"),
        "Decision Tree": DecisionTreeClassifier(featuresCol="features", labelCol="label"),
        "Random Forest": RandomForestClassifier(featuresCol="features", labelCol="label"),
        "Gradient Boosted Trees": GBTClassifier(featuresCol="features", labelCol="label")
    }

    param_grids = {
        "Logistic Regression": ParamGridBuilder().addGrid(models["Logistic Regression"].regParam, [0.01, 0.1]).build(),
        "Decision Tree": ParamGridBuilder().addGrid(models["Decision Tree"].maxDepth, [3, 5, 7]).build(),
        "Random Forest": ParamGridBuilder().addGrid(models["Random Forest"].numTrees, [10, 20]).build(),
        "Gradient Boosted Trees": ParamGridBuilder().addGrid(models["Gradient Boosted Trees"].maxIter, [10, 20]).build()
    }

    for name, model in models.items():
        print(f"\n🔍 Tuning {name}...")
        cv = CrossValidator(estimator=model,
                            estimatorParamMaps=param_grids[name],
                            evaluator=evaluator,
                            numFolds=5)
        cv_model = cv.fit(train_df)
        predictions = cv_model.transform(test_df)
        auc = evaluator.evaluate(predictions)
        print(f"✅ {name} Best AUC: {auc:.4f}")
        print(f"🏷️  Best Params: {cv_model.bestModel.extractParamMap()}")

# --- Run all tasks ---
preprocessed_df = preprocess_data(df)
train_logistic_regression_model(preprocessed_df)
feature_selection(preprocessed_df)
tune_and_compare_models(preprocessed_df)

# Stop Spark session
spark.stop()