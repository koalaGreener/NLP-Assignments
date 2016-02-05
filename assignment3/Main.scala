package uk.ac.ucl.cs.mr.statnlpbook.assignment3

/**
  * @author rockt
  */
object Main extends App {
  /**
    * Example training of a model
    *
    * Problems 2/3/4: perform a grid search over the parameters below
    */
  val learningRate = 0.0001
  val vectorRegularizationStrength = 0.2
  val matrixRegularizationStrength = 0.2


  val wordDim = 10
  val hiddenDim = 10

  val trainSetName = "train"
  val validationSetName = "dev"

  //val model: Model = new SumOfWordVectorsModel(wordDim, vectorRegularizationStrength)
  val model: Model = new RecurrentNeuralNetworkModel(wordDim, hiddenDim, vectorRegularizationStrength, matrixRegularizationStrength)
  //val model: Model = new LongShortTermMemoryNetworks(wordDim, hiddenDim, vectorRegularizationStrength, matrixRegularizationStrength)




  def epochHook(iter: Int, accLoss: Double): Unit = {
    println("Epoch %4d\tLoss %8.4f\tTrain Acc %4.2f\tDev Acc %4.2f".format(
      iter, accLoss, 100 * Evaluator(model, trainSetName), 100 * Evaluator(model, validationSetName)))
  }

  StochasticGradientDescentLearner(model, trainSetName, 500, learningRate, epochHook)



}