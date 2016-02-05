package uk.ac.ucl.cs.mr.statnlpbook.assignment1

import java.io.File

import uk.ac.ucl.cs.mr.statnlpbook.chapter.languagemodels.{LanguageModel, Util}

import scala.collection.mutable

/**
 * @author riedel
 */
object OpenEndedLMQuestion {

  /**
   * An LM that performs well on the dev set.
   */
  // pp = 336
  case class MyReallyGooDLM(vocab: Set[String],trainFile: File) extends LanguageModel {
    def order = 20
    val alpha = 25
    val vsize = 30
    // being calculated in object DataAnalysis
    val Vforwords = 7314
    val countofwords = 46305
    val beta = 1.6
    val train = Assignment1Util.loadWords(trainFile)
    var result = scala.collection.mutable.ListBuffer[Int]()
    var counts = 0
    for(i <- train)
    {
      if(i.equals("[BAR]"))
        counts = 0
      else if(i.equals("[/BAR]"))
        result += counts
      else
        counts += 1
    }
    val op = result.groupBy(w => w).mapValues(_.size)
    val sortedop = scala.collection.mutable.Map(op.toSeq.sortBy(_._1):_*)
    sortedop += (28 -> 0)
    val sortedop2 = mutable.ListMap(sortedop.toSeq.sortBy(_._1):_*)
    val lengthAndprobability = scala.collection.mutable.Map[Int, Double]()
    val SumBAR = 5611
    def subSum(a:Int):Int = {
      var sub = 0
      var count = a
      while(count > 0){
        sub += sortedop2(count-1)
        count = count - 1
      }
      return sub
    }

    //realtion between distance and counts
    for(k <- sortedop2)
    {
      if(k._1 < vsize)
        lengthAndprobability.update(k._1, (k._2 + alpha)/((SumBAR - subSum(k._1)) + 1.0 * alpha * vsize))
    }

    //count the times that each word appears
    val countsasamap = new scala.collection.mutable.HashMap[String,Int] withDefaultValue 0
    for (i <- train)
      countsasamap(i) = countsasamap(i) + 1

    //TODO: This needs to be improved by you.
    def probability(word: String, history: String*): Double ={
      val totalElements = 7313.0
      var index = 0
      var distance = 0
      var boolean = true
      for(i <- (history.length - 1) until -1 by -1 ){ //19 elements
        if (history(i).equals("[BAR]") & boolean) {
          distance = index
          boolean = false
        }
        else
          index = index + 1
      }
      //println(lengthAndprobability)
      if(history.length > 1 && history(history.length - 1).equals("[/BAR]"))
        return 1.0
      else if (word.equals("[/BAR]"))
        return lengthAndprobability(distance)
      else
        return (1 - lengthAndprobability(distance)) * ( 1.0 * (countsasamap(word) + beta) / (countofwords + Vforwords * beta ))



    }

  }



  def main(args: Array[String]) {
    //The training file we provide you
    val trainFile = new File(args(0))

    //the dev file we provide you.
    val devFile = new File(args(1))

    //the training sequence of words
    val train = Assignment1Util.loadWords(trainFile).toBuffer

    //the dev sequence of words
    val dev = Assignment1Util.loadWords(devFile).toBuffer

    //the vocabulary. Contains the training words and the OOV symbol (as the dev set has been preprocessed by
    //replacing words not in the training set with OOV).
    val vocab = train.toSet + Util.OOV

    //TODO: Improve the MyBarAwareLM implementation

    //This calculates the perplexity of the
    val lm = MyReallyGooDLM(vocab, trainFile)
    val pp = LanguageModel.perplexity(lm, dev)
    println(pp)

    // Test for the parameter
/*    for (i<-0.0 to 1.0 by 0.1)
      {
        val lm6 = InterpolatedLM(lm, NGramLM(train.toIndexedSeq, 1), i)
        val pp2 = LanguageModel.perplexity(lm6, dev)
        println(, pp2)
      }*/

  }

}
