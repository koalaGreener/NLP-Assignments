package uk.ac.ucl.cs.mr.statnlpbook.assignment1

import java.io.File

import scala.collection.immutable.ListMap

/**
 * @author riedel
 */
object DataAnalysis {

  /**
   * An LM that assigns increasing probability to the [/BAR] token the further the last [/BAR] token is away,
   * and uniform probability for all other tokens. How to choose the function that maps [/BAR] distance to a probability
   * is left to you, but you should guide your choice by the goal of minimizing perplexity.
   * @param vocab the vocabulary.
   */


  def main(args: Array[String]) {
    //The training file we provide you
    val trainFile = new File(args(0))

    //the training sequence of words/List
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
    //println(op)
    val sortedop = scala.collection.mutable.Map(op.toSeq.sortBy(_._1):_*)
    sortedop += (28 -> 0)
    val sortedop2 = ListMap(sortedop.toSeq.sortBy(_._1):_*)
    //println(sortedop2)

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
        //println(k)
        if(k._1 < 30)
        lengthAndprobability.update(k._1, k._2*1.0/(SumBAR - subSum(k._1)))
        //println(sortedop)
      }


    //realtion between distance and probability
    val ResultOfRate = ListMap(lengthAndprobability.toSeq.sortBy(_._1):_*)
    //println(ResultOfRate)


    //[BAR] pairs = 5611
    var sum = 0
    for(i <- op)
        sum = sum + i._2
    //println(sum)


    // |V|=7314   count of words = 46305
    var outputList = List[String]()
    var outputList2 = List[String]()
    outputList = train
    //println(outputList.length - sum *2)
    outputList2 = train.distinct
    //println(outputList2.length)


  }
}
