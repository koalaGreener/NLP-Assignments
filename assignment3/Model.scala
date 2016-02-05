package uk.ac.ucl.cs.mr.statnlpbook.assignment3


import breeze.linalg.DenseVector

import scala.collection.mutable

/**
 * @author rockt
 */
trait Model {
  /**
   * Stores all vector parameters
   */
  val vectorParams = new mutable.HashMap[String, VectorParam]()
  /**
   * Stores all matrix parameters
   */
  val matrixParams = new mutable.HashMap[String, MatrixParam]()
  /**
   * Maps a word to its trainable or fixed vector representation
   * @param word the input word represented as string
   * @return a block that evaluates to a vector/embedding for that word
   */
  def wordToVector(word: String): Block[Vector]
  /**
   * Composes a sequence of word vectors to a sentence vectors
   * @param words a sequence of blocks that evaluate to word vectors
   * @return a block evaluating to a sentence vector
   */
  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector]
  /**
   * Calculates the score of a sentence based on the vector representation of that sentence
   * @param sentence a block evaluating to a sentence vector
   * @return a block evaluating to the score between 0.0 and 1.0 of that sentence (1.0 positive sentiment, 0.0 negative sentiment)
   */
  def scoreSentence(sentence: Block[Vector]): Block[Double]
  /**
   * Predicts whether a sentence is of positive or negative sentiment (true: positive, false: negative)
   * @param sentence a tweet as a sequence of words
   * @param threshold the value above which we predict positive sentiment
   * @return whether the sentence is of positive sentiment
   */
  def predict(sentence: Seq[String])(implicit threshold: Double = 0.5): Boolean = {
    val wordVectors = sentence.map(wordToVector)
    val sentenceVector = wordVectorsToSentenceVector(wordVectors)
    scoreSentence(sentenceVector).forward() >= threshold
  }
  /**
   * Defines the training loss
   * @param sentence a tweet as a sequence of words
   * @param target the gold label of the tweet (true: positive sentiement, false: negative sentiment)
   * @return a block evaluating to the negative log-likelihod plus a regularization term
   */
  def loss(sentence: Seq[String], target: Boolean): Loss = {
    val targetScore = if (target) 1.0 else 0.0
    val wordVectors = sentence.map(wordToVector)
    val sentenceVector = wordVectorsToSentenceVector(wordVectors)
    val score = scoreSentence(sentenceVector)
    new LossSum(NegativeLogLikelihoodLoss(score, targetScore), regularizer(wordVectors))
  }
  /**
   * Regularizes the parameters of the model for a given input example
   * @param words a sequence of blocks evaluating to word vectors
   * @return a block representing the regularization loss on the parameters of the model
   */
  def regularizer(words: Seq[Block[Vector]]): Loss
}


/**
 * Problem 2
 * A sum of word vectors model
 * @param embeddingSize dimension of the word vectors used in this model  多少维
 * @param regularizationStrength strength of the regularization on the word vectors and global parameter vector w
 */
class SumOfWordVectorsModel(embeddingSize: Int, regularizationStrength: Double) extends Model {
  /**
   * We use a lookup table to keep track of the word representations
   */
  override val vectorParams: mutable.HashMap[String, VectorParam] = LookupTable.trainableWordVectors
  /**
   * We are also going to need another global vector parameter
   */
  vectorParams += "param_w" -> VectorParam(embeddingSize)

  def wordToVector(word: String): Block[Vector] = LookupTable.addTrainableWordVector(word, embeddingSize)

  //too fucking difficult in this part
  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector] = {
    Sum(words)
  }

  def scoreSentence(sentence: Block[Vector]): Block[Double] = {
    Sigmoid(Dot(vectorParams("param_w"), sentence))
  }

  def regularizer(words: Seq[Block[Vector]]): Loss = {
    L2Regularization(regularizationStrength, words :+ vectorParams("param_w"):_*)
  }
}


/**
  * Problem 3
  * A recurrent neural network model
  * @param embeddingSize dimension of the word vectors used in this model
  * @param hiddenSize dimension of the hidden state vector used in this model
  * @param vectorRegularizationStrength strength of the regularization on the word vectors and global parameter vector w
  * @param matrixRegularizationStrength strength of the regularization of the transition matrices used in this model
  */
class RecurrentNeuralNetworkModel(embeddingSize: Int, hiddenSize: Int,
                                  vectorRegularizationStrength: Double,
                                  matrixRegularizationStrength: Double) extends Model {
  override val vectorParams: mutable.HashMap[String, VectorParam] = LookupTable.trainableWordVectors
  //
  vectorParams += "param_w" -> VectorParam(embeddingSize)
  //
  vectorParams += "param_h0" -> VectorParam(hiddenSize)
  //
  vectorParams += "param_b" -> VectorParam(hiddenSize)

  //override val matrixParams: mutable.HashMap[String, MatrixParam] = LookupTable.trainableWordMatrixs
  override val matrixParams: mutable.HashMap[String, MatrixParam] = new mutable.HashMap[String, MatrixParam]()

  matrixParams += "param_Wx" -> MatrixParam(hiddenSize, embeddingSize)
  matrixParams += "param_Wh" -> MatrixParam(hiddenSize, hiddenSize)

  def wordToVector(word: String): Block[Vector] = LookupTable.addTrainableWordVector(word, embeddingSize)

  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector] = {
    words.foldLeft(vectorParams("param_h0"): Block[Vector]) { (z: Block[Vector], i: Block[Vector]) => Tanh(Sum(mutable.Seq(Mul(matrixParams("param_Wh"), z), Mul(matrixParams("param_Wx"), i), vectorParams("param_b")))) }
    //LookupTable.get("param_h0")
  }

  def scoreSentence(sentence: Block[Vector]): Block[Double] = Sigmoid(Dot(vectorParams("param_w"), sentence))

  def regularizer(words: Seq[Block[Vector]]): Loss =
    new LossSum(
      L2Regularization(vectorRegularizationStrength, words:+ vectorParams("param_w") :+ vectorParams("param_h0") :+ vectorParams("param_b"):_*),
      L2Regularization(matrixRegularizationStrength, words:+ matrixParams("param_Wx"):+ matrixParams("param_Wh"):_*)
    )
}


///**
//  * Problem 4
//  * LSTM
//  * @param embeddingSize dimension of the word vectors used in this model
//  * @param hiddenSize dimension of the hidden state vector used in this model
//  * @param vectorRegularizationStrength strength of the regularization on the word vectors and global parameter vector w
//  * @param matrixRegularizationStrength strength of the regularization of the transition matrices used in this model
//
//class LongShortTermMemoryNetworks(embeddingSize: Int, hiddenSize: Int,
//                                  vectorRegularizationStrength: Double,
//                                  matrixRegularizationStrength: Double) extends Model {
//  override val vectorParams: mutable.HashMap[String, VectorParam] = LookupTable.trainableWordVectors
//  //Xt
//  vectorParams += "param_w" -> VectorParam(embeddingSize)
//  //Ht
//  vectorParams += "param_h0" -> VectorParam(hiddenSize)
//  //Ct
//  vectorParams += "param_c0" -> VectorParam(hiddenSize)
//  //b
//  vectorParams += "param_bf" -> VectorParam(hiddenSize)
//  vectorParams += "param_bi" -> VectorParam(hiddenSize)
//  vectorParams += "param_bc" -> VectorParam(hiddenSize)
//  vectorParams += "param_bo" -> VectorParam(hiddenSize)
//
//
//  override val matrixParams: mutable.HashMap[String, MatrixParam] = new mutable.HashMap[String, MatrixParam]()
//  //W
//  matrixParams += "param_Wzh" -> MatrixParam(hiddenSize, hiddenSize)
//  matrixParams += "param_Wzx" -> MatrixParam(hiddenSize, embeddingSize)
//
//  matrixParams += "param_Wrh" -> MatrixParam(hiddenSize, hiddenSize)
//  matrixParams += "param_Wrx" -> MatrixParam(hiddenSize, embeddingSize)
//
//  matrixParams += "param_Whh" -> MatrixParam(hiddenSize, hiddenSize)
//  matrixParams += "param_Whx" -> MatrixParam(hiddenSize, embeddingSize)
//
//  matrixParams += "param_Woh" -> MatrixParam(hiddenSize, hiddenSize)
//  matrixParams += "param_Wox" -> MatrixParam(hiddenSize, embeddingSize)
//
//
//  def wordToVector(word: String): Block[Vector] = LookupTable.addTrainableWordVector(word,embeddingSize)
//
//
//
//  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector] = {
//
//    /* LSTM
//        var Ft = Sigmoid(Sum(mutable.Seq(Mul(matrixParams("param_Wfh"),vectorParams("param_h0")) + Mul(matrixParams("param_Wfx"),vectorParams("param_w")) + vectorParams("param_bf"))))
//        var it = Sigmoid(Mul(matrixParams("param_Wih"),vectorParams("param_h0")) + Mul(matrixParams("param_Wix"),vectorParams("param_w")) + vectorParams("param_bi"))
//        var ct = Tanh(Mul(matrixParams("param_Wch"),vectorParams("param_h0")) + Mul(matrixParams("param_Wix"),vectorParams("param_w")) + vectorParams("param_bc"))
//        var ctt = Ft * ctt + it * ct
//        var Ot = Sigmoid(Mul(matrixParams("param_Woh"),vectorParams("param_h0")) + Mul(matrixParams("param_Wox"),vectorParams("param_w")) + vectorParams("param_bo"))
//        var ht = Ot * Tanh(ctt)
//
//
//    //GRU
//    var Zt = SigmoidBlock(Sum(mutable.Seq( Mul(matrixParams("param_Wzh") , vectorParams("param_h0")) , Mul(matrixParams("param_Wzx"), vectorParams("param_w")))))
//    var Rt = SigmoidBlock(Sum(mutable.Seq(Mul(matrixParams("param_Wrh"), vectorParams("param_h0")), Mul(matrixParams("param_Wrx"),  vectorParams("param_w")))))
//    var ParameterHt = Tanh(Sum(mutable.Seq(Mul(matrixParams("param_Whh"), (Rt:*vectorParams("param_h0"))), Mul(matrixParams("param_Whx"), vectorParams("param_w")))))
//    var oneVector = LookupTable.addTrainableWordVector("param_one", DenseVector.ones[Double](embeddingSize))
//    //var Ht = Sum(mutable.Seq((vectorParams("param_h0"):*(oneVector:-Zt)), Zt:*ParameterHt))
//    words.foldLeft(vectorParams("param_h0"):Block[Vector]){(z:Block[Vector],i:Block[Vector])=>Sum(mutable.Seq((vectorParams("param_h0"):*(oneVector:-Zt)), Zt:*ParameterHt))}
//
//
//
//    val paramOne: Vector = DenseVector.ones[Double](embeddingSize)
//    words.foldLeft(vectorParams("param_h0"):Block[Vector]){(z:Block[Vector],i:Block[Vector])=>Sum(mutable.Seq(Hadamard(Sum(mutable.Seq(paramOne,Abstract(mutable.Seq(SigmoidBlock(Sum(mutable.Seq(Mul(matrixParams("param_Wfh"),z),Mul(matrixParams("param_Wfx"),i)))))))),z),Hadamard(SigmoidBlock(Sum(mutable.Seq(Mul(matrixParams("param_Wfh"),z),Mul(matrixParams("param_Wfx"),i)))),Tanh(Sum(mutable.Seq(Mul(matrixParams("param_Wch"),Hadamard(SigmoidBlock(Sum(mutable.Seq(Mul(matrixParams("param_Wih"),z),Mul(matrixParams("param_Wix"),i)))),z)),Mul(matrixParams("param_Wcx"),i)))))))}
//    //1:Hadamard(Sum(mutable.Seq(paramOne,Abstract(mutable.Seq(SigmoidBlock(Sum(mutable.Seq(Mul(matrixParams("param_Wfh"),z),Mul(matrixParams("param_Wfx"),i)))))))),z)
//    //2:SigmoidBlock(Sum(mutable.Seq(Mul(matrixParams("param_Wfh"),z),Mul(matrixParams("param_Wfx"),i))))
//    //3:Tanh(Sum(mutable.Seq(Mul(matrixParams("param_Wch"),Hadamard(SigmoidBlock(Sum(mutable.Seq(Mul(matrixParams("param_Wih"),z),Mul(matrixParams("param_Wix"),i)))),z)),Mul(matrixParams("param_Wcx"),i))))}
//    ???
//  }
//
//
//
//
//  def scoreSentence(sentence: Block[Vector]): Block[Double] = Sigmoid(Dot(vectorParams("param_w"),sentence))
//
//  def regularizer(words: Seq[Block[Vector]]): Loss =
//    new LossSum(
//      L2Regularization(vectorRegularizationStrength, words:+ vectorParams("param_w") :+ vectorParams("param_h0") :+ vectorParams("param_b"):_*),
//      L2Regularization(matrixRegularizationStrength, words:+ matrixParams("param_Wh"):+ matrixParams("param_Wx"):_*)
//    )
//
//
//}*/*/


/**
  * Problem 4
  * LSTM
  * @param embeddingSize dimension of the word vectors used in this model
  * @param hiddenSize dimension of the hidden state vector used in this model
  * @param vectorRegularizationStrength strength of the regularization on the word vectors and global parameter vector w
  * @param matrixRegularizationStrength strength of the regularization of the transition matrices used in this model
  */

class LongShortTermMemoryNetworks(embeddingSize: Int, hiddenSize: Int,
                                  vectorRegularizationStrength: Double = 0.0,
                                  matrixRegularizationStrength: Double = 0.0) extends Model {
  override val vectorParams: mutable.HashMap[String, VectorParam] = LookupTable.trainableWordVectors
  //Xt
  vectorParams += "param_w" -> VectorParam(embeddingSize)
  //Ht
  vectorParams += "param_h0" -> VectorParam(hiddenSize)


  override val matrixParams: mutable.HashMap[String, MatrixParam] = new mutable.HashMap[String, MatrixParam]()
  //W
  matrixParams += "param_Wzh" -> MatrixParam(hiddenSize, hiddenSize)
  matrixParams += "param_Wzx" -> MatrixParam(hiddenSize, embeddingSize)

  matrixParams += "param_Wrh" -> MatrixParam(hiddenSize, hiddenSize)
  matrixParams += "param_Wrx" -> MatrixParam(hiddenSize, embeddingSize)

  matrixParams += "param_Whh" -> MatrixParam(hiddenSize, hiddenSize)
  matrixParams += "param_Whx" -> MatrixParam(hiddenSize, embeddingSize)



  def wordToVector(word: String): Block[Vector] = LookupTable.addTrainableWordVector(word,embeddingSize)

  def wordVectorsToSentenceVector(words: Seq[Block[Vector]]): Block[Vector] = {
    //val paramOne: Vector = DenseVector.ones[Double](hiddenSize)
    //words.foldLeft(vectorParams("param_h0"):Block[Vector]){(z:Block[Vector],i:Block[Vector])=>Sum(mutable.Seq(Hadamard(Sum(mutable.Seq(paramOne,Minus(mutable.Seq(SigmoidBlock(Sum(mutable.Seq(Mul(matrixParams("param_Wfh"),z),Mul(matrixParams("param_Wfx"),i)))))))),z),Hadamard(SigmoidBlock(Sum(mutable.Seq(Mul(matrixParams("param_Wfh"),z),Mul(matrixParams("param_Wfx"),i)))),Tanh(Sum(mutable.Seq(Mul(matrixParams("param_Wch"),Hadamard(SigmoidBlock(Sum(mutable.Seq(Mul(matrixParams("param_Wih"),z),Mul(matrixParams("param_Wix"),i)))),z)),Mul(matrixParams("param_Wcx"),i)))))))}

    //1:Hadamard(Sum(mutable.Seq(paramOne,Minus(mutable.Seq(SigmoidBlock(Sum(mutable.Seq(Mul(matrixParams("param_Wfh"),z),Mul(matrixParams("param_Wfx"),i)))))))),z)
    //2:SigmoidBlock(Sum(mutable.Seq(Mul(matrixParams("param_Wfh"),z),Mul(matrixParams("param_Wfx"),i))))
    //3:Tanh(Sum(mutable.Seq(Mul(matrixParams("param_Wch"),Hadamard(SigmoidBlock(Sum(mutable.Seq(Mul(matrixParams("param_Wih"),z),Mul(matrixParams("param_Wix"),i)))),z)),Mul(matrixParams("param_Wcx"),i))))}

    //    //GRU
        /*var Zt = SigmoidBlock(Sum(mutable.Seq( Mul(matrixParams("param_Wzh") , vectorParams("param_h0")) , Mul(matrixParams("param_Wzx"), vectorParams("param_w")))))
        var Rt = SigmoidBlock(Sum(mutable.Seq(Mul(matrixParams("param_Wrh"), vectorParams("param_h0")), Mul(matrixParams("param_Wrx"),  vectorParams("param_w")))))
        var ParameterHt = Tanh(Sum(mutable.Seq(Mul(matrixParams("param_Whh"), Hadamard(Rt, vectorParams("param_h0"))), Mul(matrixParams("param_Whx"), vectorParams("param_w")))))
        var oneVector = LookupTable.addTrainableWordVector("param_one", DenseVector.ones[Double](embeddingSize))
        words.foldLeft(vectorParams("param_h0"):Block[Vector]){(z:Block[Vector],i:Block[Vector])=>Sum(mutable.Seq(Hadamard(Minus(vectorParams("param_one"),Zt),z), Hadamard(Zt,ParameterHt)))}*/

        //var Zt = SigmoidBlock(Sum(mutable.Seq( Mul(matrixParams("param_Wzh") , z) , Mul(matrixParams("param_Wzx"), vectorParams("param_w")))))
        //var Rt = SigmoidBlock(Sum(mutable.Seq(Mul(matrixParams("param_Wrh"), z), Mul(matrixParams("param_Wrx"),  vectorParams("param_w")))))
        //var ParameterHt = Tanh(Sum(mutable.Seq(Mul(matrixParams("param_Whh"), Hadamard(SigmoidBlock(Sum(mutable.Seq(Mul(matrixParams("param_Wrh"), z), Mul(matrixParams("param_Wrx"),  vectorParams("param_w"))))), z)), Mul(matrixParams("param_Whx"), vectorParams("param_w")))))
        var oneVector = LookupTable.addTrainableWordVector("param_one", DenseVector.ones[Double](embeddingSize))
        words.foldLeft(vectorParams("param_h0"):Block[Vector]){(z:Block[Vector],i:Block[Vector])=>Sum(mutable.Seq(Hadamard(Minus(vectorParams("param_one"),SigmoidBlock(Sum(mutable.Seq( Mul(matrixParams("param_Wzh") , z) , Mul(matrixParams("param_Wzx"), vectorParams("param_w")))))),z), Hadamard(SigmoidBlock(Sum(mutable.Seq( Mul(matrixParams("param_Wzh") , z) , Mul(matrixParams("param_Wzx"), vectorParams("param_w"))))),Tanh(Sum(mutable.Seq(Mul(matrixParams("param_Whh"), Hadamard(SigmoidBlock(Sum(mutable.Seq(Mul(matrixParams("param_Wrh"), z), Mul(matrixParams("param_Wrx"),  vectorParams("param_w"))))), z)), Mul(matrixParams("param_Whx"), vectorParams("param_w"))))))))}

  }

  def scoreSentence(sentence: Block[Vector]): Block[Double] = Sigmoid(Dot(vectorParams("param_w"),sentence))

  def regularizer(words: Seq[Block[Vector]]): Loss =
    new LossSum(
      L2Regularization(vectorRegularizationStrength, words :+ vectorParams("param_w") :+ vectorParams("param_h0"):_*),
      L2Regularization(matrixRegularizationStrength, words :+ matrixParams("param_Wzx"):+ matrixParams("param_Wzh"):+ matrixParams("param_Wrx"):+ matrixParams("param_Wrh"):+ matrixParams("param_Whx"):+ matrixParams("param_Whh"):_*)
    )


}



