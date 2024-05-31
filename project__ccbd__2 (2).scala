// Databricks notebook source
//closed_form solution
import org.apache.spark.ml.linalg.{DenseVector, Vectors}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.sql.SparkSession
import breeze.linalg.{pinv, DenseMatrix => BreezeDenseMatrix, DenseVector => BreezeDenseVector}



def closedFormSolution(X: RowMatrix, y: DenseVector): DenseVector = {
  val XtransX = X.computeGramianMatrix()

  val XtXI = pinv(new BreezeDenseMatrix[Double](XtransX.numRows, XtransX.numCols, XtransX.toArray))

  val XtransY: DenseVector = {
    val tempX = new BreezeDenseMatrix[Double](X.numRows().toInt, X.numCols().toInt, X.rows.collect.flatMap(_.toArray))
    val tempY = new BreezeDenseMatrix[Double](y.size, 1, y.toArray)
    val result = tempX.t * tempY
    new DenseVector(result.toArray)
  }

  val yMtx = new BreezeDenseMatrix[Double](XtransY.size, 1, XtransY.toArray)
  val results = XtXI * yMtx 
  new DenseVector(results.toArray)
}

val input = Seq(
  LabeledPoint(1.0, Vectors.dense(1.0, 9.0,4.0)),
  LabeledPoint(2.0, Vectors.dense(11.0, 6.0,7.0)),
  LabeledPoint(4.0, Vectors.dense(90.0, 8.0,8.0))
)

val rdd = spark.sparkContext.parallelize(input)

val features = rdd.map(lp => lp.features).persist()
val y = new DenseVector(rdd.map(lp => lp.label).collect())

val rows = features.map(vec => org.apache.spark.mllib.linalg.Vectors.dense(vec.toArray))
val X = new RowMatrix(rows)


val weights = closedFormSolution(X, y)

println(" Weights:"+ weights)


// COMMAND ----------

//Bonus 
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{Vector, Vectors}
import breeze.linalg.{DenseVector => BDenseVector}

// Function to convert Breeze DenseVector to Spark Vector
def toSpark(vt: BDenseVector[Double]): Vector = {
  Vectors.dense(vt.toArray)
}

// Function to convert Spark Vector to Breeze DenseVector
def toBreeze(vt: Vector): BDenseVector[Double] = {
  BDenseVector(vt.toArray)
}

// Function to compute the summand for each data point
def compteSummand(x: Vector, y: Double, theta: Vector): Vector = {
  // Convert input vectors to Breeze DenseVector for easy mathematical operations
  val xval = toBreeze(x)
  val thetaval = toBreeze(theta)

  // Compute the prediction using dot product of x and theta
  val prediction = xval dot thetaval

  // Calculate the error between prediction and actual y
  val err = prediction - y

  // Compute the summand for gradient descent
  val summand = toSpark(xval * err)

  // Return the result as a Spark Vector
  summand
}

// Main function
def main(args: Array[String]): Unit = {
  // Initial theta values
  val theta = Vectors.dense(0.5, 0.3, -0.2)

  // Sample data points with features (x) and labels (y)
  val input = Seq(
    (Vectors.dense(5.0, 4.0, 2.0), 7.0),
    (Vectors.dense(5.0, 6.0, 2.0), 1.0)
  )

  // Iterate over each data point and compute the summand
  input.foreach { case (x, y) =>
    val Result = compteSummand(x, y, theta)
    println("Final Result: " + Result)
  }
}

// Run the main function
main(Array())


// COMMAND ----------

import org.apache.spark.rdd.RDD
import org.apache.spark.ml.linalg.DenseVector

// Function to predict the label using a weight vector (w) and a LabeledPoint observation
def FindLabes(d: DenseVector, observation: LabeledPoint): (Double, Double) = {
  // Extract features and actual label from the observation
  val features = observation.features
  val actualLabel = observation.label

  // Compute the predicted label using the dot product of the weight vector and features
  val prediction = d dot features

  // Return a tuple containing the actual label and the predicted label
  (actualLabel, prediction)
}

// Function to compute the Root Mean Squared Error (RMSE) from a list of (actual, predicted) values
def calculateRMSE(predictions: RDD[(Double, Double)]): Double = {
  // Calculate the sum of squared differences between actual and predicted labels
  val sumSquedDiff = predictions.map { case (actual, prediction) =>
    val diff = actual - prediction
    diff * diff
  }.sum()

  // Calculate the mean squared error
  val meanSqredErr = sumSquedDiff / predictions.count()

  // Calculate the root mean squared error
  val rootMeanSqredErr = scala.math.sqrt(meanSqredErr)

  // Return the RMSE value
  rootMeanSqredErr
}

// Initial weight vector
val d = new DenseVector(Array(0.8, 0.5, -0.3))

// Sample data points with features and labels
val input = Seq(
  LabeledPoint(32.0, new DenseVector(Array(9.0, 2.0, 4.0))),
  LabeledPoint(7.0, new DenseVector(Array(9.0, 7.0, 2.0))),
  LabeledPoint(9.0, new DenseVector(Array(80.0, 8.0, 8.0)))
)

// Create an RDD from the sample data
val rdd = spark.sparkContext.parallelize(input)

// Generate predictions using the predictLabel function
val pre = rdd.map(observation => FindLabes(d, observation))

// Compute the RMSE from the predictions
val Result = calculateRMSE(pre)

// Print the RMSE value
println(s"Final RSME: $Result")


// COMMAND ----------

import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import breeze.linalg.{DenseVector => BDenseVector}

// Function to convert a Spark Vector to a Breeze DenseVector
def toBreeze(v: Vector): BDenseVector[Double] = {
  BDenseVector(v.toArray)
}

// Function to convert a Breeze DenseVector to a Spark Vector
def toSpark(v: BDenseVector[Double]): Vector = {
  Vectors.dense(v.toArray)
}

// Function to perform element-wise multiplication of two Spark Vectors
def elemntWisMulti(v1: Vector, v2: Vector): Vector = {
  // Zip the arrays, multiply corresponding elements, and create a new Vector
  val multipedArr = v1.toArray.zip(v2.toArray).map { case (x, y) => x * y }
  Vectors.dense(multipedArr)
}

// Function to compute the summand for one data point in the gradient descent algorithm
def comSummand(x: Vector, y: Double, theta: Vector): Vector = {
  // Convert Vectors to Breeze DenseVectors for numerical operations
  val xval = toBreeze(x)
  val tetaval = toBreeze(theta)

  // Compute prediction, error, and summand
  val prediction = xval dot tetaval
  val err = prediction - y
  val summand = elemntWisMulti(x, Vectors.dense(Array.fill(x.size)(err)))
  summand
}

// Function to create a LabeledPoint from features and a label
def createLabeledPoint(features: Vector, label: Double): LabeledPoint = {
  LabeledPoint(label, features)
}

// Gradient Descent algorithm for linear regression
def calculategrad(trainData: RDD[LabeledPoint], ittrnumber: Int): (Vector, Array[Double]) = {
  // Set initial learning rate, weight vector, and an array to store training errors
  var alp = 0.001
  var wi = Vectors.zeros(trainData.first().features.size)
  val traningErr = new Array[Double](ittrnumber)

  for (i <- 0 until ittrnumber) {
    // Update learning rate
    alp = alp / (math.sqrt(i + 1))

    // Compute gradients by mapping over the training data and reducing
    val gradients = trainData.map { labeledPoint =>
      val features = labeledPoint.features
      val prediction = wi dot features
      val error = prediction - labeledPoint.label
      comSummand(features, error, wi)
    }.reduce((a, b) => Vectors.dense((toBreeze(a) + toBreeze(b)).toArray))

    // Update weight vector using the learning rate and gradients
    wi = Vectors.dense((toBreeze(wi) - (toBreeze(gradients) * alp)).toArray)

    // Compute predictions, calculate RMSE, and store the training error
    val predictions = trainData.map(observation => (observation.label, wi dot observation.features))
    val rslt = calculateRMSE(predictions)
    traningErr(i) = rslt
  }

  (wi, traningErr)
}

// Function to compute Root Mean Squared Error (RMSE) from a list of (actual, predicted) values
def calculateRMSE(predictions: RDD[(Double, Double)]): Double = {
  // Calculate sum of squared differences, mean squared error, and root mean squared error
  val sumSquredDiff = predictions.map { case (actual, prediction) =>
    val d = actual - prediction
    d * d
  }.sum()

  val meanSquredErr = sumSquredDiff / predictions.count()
  val rootMeanSquredErr = math.sqrt(meanSquredErr)
  rootMeanSquredErr
}

// Sample data for linear regression
val input = Seq(
  LabeledPoint(7.0, Vectors.dense(1.0, 6.0, 8.0)),
  LabeledPoint(8.0, Vectors.dense(9.0, 9.0, 6.0)),
  LabeledPoint(4.0, Vectors.dense(6.0, 8.0, 5.0))
)

// Create an RDD from the sample data
val value = spark.sparkContext.parallelize(input)

// Number of iterations for gradient descent
val ittrnumber = 5

// Run gradient descent and get the final weights and training errors
val (weigt, trainingErrors) = calculategrad(value, ittrnumber)

// Print the final weights
println("Final Weight:" + weigt)

// Print the training error at every iteration
println("Training Error at every iteration:")
trainingErrors.reverse.foreach(println)

