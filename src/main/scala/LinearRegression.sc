import java.io.{File, PrintWriter}

import breeze.linalg.{DenseMatrix, DenseVector, csvread, pinv}
import breeze.numerics.abs
import breeze.stats.mean

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.ArrayBuffer


def MAE(y: DenseVector[Double], y_pred: DenseVector[Double]): Double = {
  mean(abs(y_pred - y)/y)
}

def downloadCSV(path: String, target: Int): Tuple2[DenseMatrix[Double], DenseVector[Double]] = {
  val csvFile: File = new File(path)
  val data: DenseMatrix[Double] = csvread(csvFile, skipLines = 1)
  val maxColNum = data.cols - 1
  val cols = for (i <- 0 to maxColNum if i != target) yield i
  val X = data(::, cols).toDenseMatrix
  val y: DenseVector[Double] = data(::, target)
  (X, y)
}

def toCSV(path: String, iterator: DenseVector[Double]): Unit = {
  val writer = new PrintWriter(new File(path))

  for (el <- iterator) {
    writer.write(f"$el\n")
  }
  writer.close()
}


class LinearRegression() {

  var w: DenseVector[Double] = DenseVector.ones[Double](0)

  def predict(X: DenseMatrix[Double]): DenseVector[Double] = {
    X * w
  }

  def mse(y: DenseVector[Double], y_preds: DenseVector[Double]): Double = {
    val diff = y.toArray zip y_preds.toArray map ( z => scala.math.pow(z._1 - z._2, 2))
    diff.sum / y.length.toDouble
  }

  def mae(y: DenseVector[Double], y_preds: DenseVector[Double]): Double = {
    val diff = y.toArray zip y_preds.toArray map ( z => scala.math.abs(z._1 - z._2))
    diff.sum / y.length.toDouble
  }

  def fit(X: DenseMatrix[Double], y: DenseVector[Double], lr: Double, epochs: Int): Unit = {
    w = DenseVector.ones[Double](X.cols)
    var y_preds = DenseVector.zeros[Double](y.length)
    var loss_best = Double.MaxValue
    var loss = Double.MaxValue
    var gradients = DenseVector.zeros[Double](X.cols)

    for (i <- 0 to epochs; if loss_best >= loss) {
      loss_best = loss
      y_preds = predict(X)
      gradients = pinv(X) * (y_preds - y)
      w -= lr * gradients
      y_preds = predict(X)
      loss = mse(y, y_preds)

      if ((i&0xff) == 0xff) {
        val loss_mae = mae(y, y_preds)
        println(s"iter: $i, mse: $loss, mae: $loss_mae, weight: $w")
      }
    }
  }

}


var DEFAULT_DATA_TRAIN="C:\\Users\\user\\Documents\\Other\\HW5\\src\\main\\scala\\train.csv"
var DEFAULT_OUT_TRAIN = "C:\\Users\\user\\Documents\\Other\\HW5\\src\\main\\scala\\out-train.csv"
var DEFAULT_DATA_TEST="C:\\Users\\user\\Documents\\Other\\HW5\\src\\main\\scala\\test.csv"
var DEFAULT_OUT_TEST = "C:\\Users\\user\\Documents\\Other\\HW5\\src\\main\\scala\\out-test.csv"

val df_train = downloadCSV(DEFAULT_DATA_TRAIN,3)
val X_train = df_train._1
val y_train = df_train._2
val df_test = downloadCSV(DEFAULT_DATA_TEST,3)
val X_test = df_test._1
val y_test = df_test._2


val model = new LinearRegression()
val lr = 0.0003
val epochs = 30000
model.fit(X_train, y_train, lr, epochs)

val y_pred_train = model.predict(X_train)
val y_pred_test = model.predict(X_test)
val mae_train = MAE(y_pred_train, y_train)
val mae_test = MAE(y_pred_test, y_test)


toCSV(DEFAULT_OUT_TRAIN, y_pred_train)
toCSV(DEFAULT_OUT_TEST, y_pred_test)