package uk.ac.ucl.cs.mr.statnlpbook.assignment3

/**
  * Problem 1
  */
object GradientChecker extends App {
  val EPSILON = 1e-6

  /**
    * For an introduction see http://cs231n.github.io/neural-networks-3/#gradcheck
    *
    * This is a basic implementation of gradient checking.
    * It is restricted in that it assumes that the function to test evaluates to a double.
    * Moreover, another constraint is that it always tests by backpropagating a gradient of 1.0.
    */
  def apply[P](model: Block[Double], paramBlock: ParamBlock[P]) = {
    paramBlock.resetGradient()
    model.forward()
    model.backward(1.0)

    var avgError = 0.0

    val gradient = paramBlock.gradParam match {
      case m: Matrix => m.toDenseVector
      case v: Vector => v
    }

    /**
      * Calculates f_theta(x_i + eps)
      * @param index i in x_i
      * @param eps value that is added to x_i
      * @return
      */
    def wiggledForward(index: Int, eps: Double): Double = {
      var result = 0.0
      paramBlock.param match {
        case v: Vector =>
          val tmp = v(index)
          v(index) = tmp + eps
          result = model.forward()
          v(index) = tmp
        case m: Matrix =>
          val (row, col) = m.rowColumnFromLinearIndex(index)
          val tmp = m(row, col)
          m(row, col) = tmp + eps
          result = model.forward()
          m(row, col) = tmp
      }
      result
    }

    for (i <- 0 until gradient.activeSize) {
      //todo: your code goes here!
      val gradientExpected: Double = (wiggledForward(i,EPSILON)-wiggledForward(i,-EPSILON))/(2*EPSILON)
      avgError = avgError + math.abs(gradientExpected - gradient(i))

      assert(
        math.abs(gradientExpected - gradient(i)) < EPSILON,
        "Gradient check failed!\n" +
          s"Expected gradient for ${i}th component in input is $gradientExpected but I got ${gradient(i)}"
      )
    }

    println("Average error: " + avgError)
  }
  /**
    * A very silly block to test if gradient checking is working.
    * Will only work if the implementation of the Dot block is already correct
    */
  //val (sentence, target) = SentimentAnalysisCorpus.getExample(corpus)
  val a = vec(-1.5, 1.0, 1.5, 0.5)
  val b = VectorParam(4)
  b.set(vec(1.0, 2.0, -0.5, 2.5))

  //val simpleBlock = Dot(a,b)
  //val simpleBlock = Sigmoid(Dot(a,b))
  //val simpleBlock = NegativeLogLikelihoodLoss(Sigmoid(Dot(a,b)),1.0)
  //val simpleBlock = L2Regularization(0.01,b)
  //GradientChecker(simpleBlock, b)

  val checkModel = new SumOfWordVectorsModel(10, 0.01)
  //val checkModel = new RecurrentNeuralNetworkModel(10, 10,0.01,0.01)
  val sentence = Seq("on", "my", "way", "home", "from", "Orlando", "-", "such", "a", "great", "time", "had")
  GradientChecker(checkModel.loss(sentence, true), checkModel.vectorParams("param_w"))

}