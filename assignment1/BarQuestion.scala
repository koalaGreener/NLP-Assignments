package uk.ac.ucl.cs.mr.statnlpbook.assignment1

import java.io.File

import uk.ac.ucl.cs.mr.statnlpbook.chapter.languagemodels.{InterpolatedLM, LanguageModel, NGramLM, Util}

import scala.collection.immutable.ListMap
/**
 * @author riedel
 */
object BarQuestion {

  /**
   * An LM that assigns increasing probability to the [/BAR] token the further the last [/BAR] token is away,
   * and uniform probability for all other tokens. How to choose the function that maps [/BAR] distance to a probability
   * is left to you, but you should guide your choice by the goal of minimizing perplexity.
   * @param vocab the vocabulary.
   */
  // Q2.2 2.3 pp = 4283 Linearly BA Model
  // 3696-3766
  case class MyBarAwareLM(vocab: Set[String]) extends LanguageModel {
    def order = 20

    //TODO: This needs to be improved by you.
    def probability(word: String, history: String*): Double ={
/*      val alpha = 0.0193
      val beta = 0.0028*/
      val alpha = 0.00649
      val beta = 0.07823
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

      if (word.equals("[/BAR]"))
        return alpha * distance + beta
      else
        return (1 - (alpha * distance + beta))/ totalElements

    }
  }
  //Q2.4 2.5 pp  =  4215 Per Distance model
  case class PerDistanceBarAwareLM(vocab: Set[String]) extends LanguageModel {
    def order = 20

    val exactProbability = Map(0 -> 0.07823917305293174, 1 -> 0.008507347254447023, 2 -> 0.0124804992199688, 3 -> 0.03771721958925751, 4 -> 0.07100348861071208, 5 -> 0.05743317870554451, 6 -> 0.07827513475509726, 7 -> 0.12026442918891432, 8 -> 0.15173410404624277,
      9 -> 0.21158432708688246, 10 -> 0.2847882454624028, 11 -> 0.34924471299093657, 12 -> 0.3519034354688951, 13 -> 0.3825214899713467, 14 -> 0.37354988399071926, 15 -> 0.3814814814814815, 16 -> 0.32934131736526945, 17 -> 0.35714285714285715, 18 -> 0.2916666666666667, 19 -> 0.35294117647058826)

    //TODO: This needs to be improved by you.
    def probability(word: String, history: String*): Double ={
      val totalElements = 7313.0
      var index = 0
      var distance = order - 1
      var boolean = true
      for(i <- (history.length - 1) until -1 by -1 ){ //19 elements
        if (history(i).equals("[BAR]") & boolean) {
          distance = index
          boolean = false
        }
        else
          index = index + 1
      }

      if (word.equals("[/BAR]"))
        return exactProbability(distance)
      else
        return (1 - exactProbability(distance))/ totalElements

    }
  }
  //Q 2.3 pp = 7313  Uniform model
  case class RegularUniformLM(vocab: Set[String]) extends LanguageModel {
    def order = 20

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
        return 1 / totalElements


    }
  }
  // Add One language model pp=4198 alpha = 25
  case class AddoneLM(vocab: Set[String],trainFile:File) extends LanguageModel {
    def order = 20
    val alpha = 25
    val vsize = 20
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
    val sortedop2 = ListMap(sortedop.toSeq.sortBy(_._1):_*)
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

    //println(lengthAndprobability)

    //TODO: This needs to be improved by you.
    def probability(word: String, history: String*): Double ={
      val totalElements = 7313.0
      var index = 0
      var distance = 19
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
/*      if(history(history.length - 1).equals("[/BAR]"))
        return 1.0
      else*/ if (word.equals("[/BAR]"))
        return lengthAndprobability(distance)
      else
        return (1 - lengthAndprobability(distance)) / totalElements


    }
  }

  // Add One language model pp=1796 alpha = 25
  case class AddoneLMImprove(vocab: Set[String],trainFile:File) extends LanguageModel {
    def order = 20
    val alpha = 25
    val vsize = 30
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
    val sortedop2 = ListMap(sortedop.toSeq.sortBy(_._1):_*)
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

    //println(lengthAndprobability)

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
        return (1 - lengthAndprobability(distance))/ totalElements


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
    val lm = MyBarAwareLM(vocab)
    val lm2 = PerDistanceBarAwareLM(vocab)
    val lm3 = RegularUniformLM(vocab)
    val lm4 = AddoneLM(vocab,trainFile)
    val lm5 = AddoneLMImprove(vocab,trainFile)
    val lm6 = InterpolatedLM(lm5, NGramLM(train.toIndexedSeq, 1), 0.3)

    //6.584*10-3 0.006584
/*    for (i <- 0.004 to 0.007 by 0.0001){
      val lm = MyBarAwareLM(vocab, i)
      val pp = LanguageModel.perplexity(lm, dev)
      //print(i + ";" )
      //print(pp + ";")
    }*/


    //This calculates the perplexity of the LM
    val pp = LanguageModel.perplexity(lm, dev)
    val pp2 = LanguageModel.perplexity(lm2, dev)
    val pp3 = LanguageModel.perplexity(lm3, dev)
    val pp4 = LanguageModel.perplexity(lm4, dev)
    val pp5 = LanguageModel.perplexity(lm5, dev)
    val pp6 = LanguageModel.perplexity(lm6, dev)


    println(pp," Linearly BA Model")
    println("---------------")
    println(pp2," Per Distance model")
    println("---------------")
    println(pp3, " Uniform model")
    println("---------------")
    println(pp4, " Add one model")
    println("---------------")
    println(pp5, " Add one model+ 2.6 Improve")
    println("---------------")
    println(pp6, " Add one model+ 2.6 Improve + Unigram LM")
    println("---------------")

/*    // 2.6 histogram of the bar length distribution
    val sampledoc = uk.ac.ucl.cs.mr.statnlpbook.chapter.languagemodels.LanguageModel.sample(lm6,Nil,1000)
    var distance = 0
    var result = scala.collection.mutable.ListBuffer[Int]()
    var counts = 0
    for(i <- sampledoc)
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
    val sortedop2 = ListMap(sortedop.toSeq.sortBy(_._1):_*)
    println(sortedop2)

    val Xaxis = ArrayBuffer[Int]()
    val Yaxis = ArrayBuffer[Int]()
    for (i <- 0 to sortedop2.size){
      if(sortedop2.contains(i)){
        Xaxis += i
        Yaxis += sortedop2(i)

      }
      else {
        Xaxis += i
        Yaxis += 0
      }
    }

    val data2 = for (i <- 0 to sortedop2.size) yield (Xaxis(i),Yaxis(i))
    val chart2 = BarChart(data2)
    chart2.title="2.6 histogram of the bar length distribution"
    chart2.show()*/



    //TODO:

    //TODO: combine a unigram model with the BAR aware LM through interpolation.
    //TODO: Improve the BarAwareLM to give probability 1 to a [BAR] following a [/BAR], and 0 otherwise.

  }
}
