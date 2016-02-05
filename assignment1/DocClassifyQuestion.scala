package uk.ac.ucl.cs.mr.statnlpbook.assignment1

import java.io.File

import uk.ac.ucl.cs.mr.statnlpbook.assignment1.Assignment1Util.Instance
import uk.ac.ucl.cs.mr.statnlpbook.chapter.languagemodels.CountLM

import scala.collection.mutable

/**
 * @author mbosnjak
 */
object DocClassifyQuestion {

  def main(args: Array[String]): Unit = {
    // load the datasets

    val train = Assignment1Util.loadDataset(new File(args(0)))
    val dev = Assignment1Util.loadDataset(new File(args(1)))
    val chosenone = train

    //calculate the priors of author (38,69,139)
    var DocCountsForJlive = 0
    var DocCountsForrakim = 0
    var DocCountsForRoots = 0

    for (instanceDoc <- chosenone.toList) {
      if (instanceDoc.author.toString.equals("Some(j_live)"))
        DocCountsForJlive = DocCountsForJlive + 1
      else if (instanceDoc.author.toString.equals("Some(rakim)"))
        DocCountsForrakim = DocCountsForrakim + 1
      else
        DocCountsForRoots = DocCountsForRoots + 1
    }
    val periodOfJlive = 1.0 * DocCountsForJlive / (DocCountsForJlive + DocCountsForrakim + DocCountsForRoots)
    val periodOfRakim = 1.0 * DocCountsForrakim / (DocCountsForJlive + DocCountsForrakim + DocCountsForRoots)
    val periodOfroots = 1.0 * DocCountsForRoots / (DocCountsForJlive + DocCountsForrakim + DocCountsForRoots)

    //calculate the |V| = 15391
    var listOfAll = scala.collection.mutable.Set[String]()
    //println(train.toList)
    for (instanceDoc <- chosenone) {
      listOfAll = listOfAll union instanceDoc.lyrics.split(" ").toSet
    }

    //calculate the count(c) 34332 54475 107144
    var countcForJlive = scala.collection.mutable.ListBuffer[String]()
    var countcForrakim = scala.collection.mutable.ListBuffer[String]()
    var countcForroots = scala.collection.mutable.ListBuffer[String]()

    for (instanceDoc <- chosenone) {
      if (instanceDoc.author.toString.equals("Some(j_live)"))
        for(elementofblank <- instanceDoc.lyrics.split(" "))
        {
          countcForJlive += elementofblank
        }
      else if (instanceDoc.author.toString.equals("Some(rakim)"))
        for(elementofblank <- instanceDoc.lyrics.split(" "))
        {
          countcForrakim += elementofblank
        }
      else
        for(elementofblank <- instanceDoc.lyrics.split(" "))
        {
          countcForroots += elementofblank
        }

    }

    //count count(w,c)
    val countsForJlive = new scala.collection.mutable.HashMap[String,Int] withDefaultValue 0
    val countsForrakim = new scala.collection.mutable.HashMap[String,Int] withDefaultValue 0
    val countsForroots = new scala.collection.mutable.HashMap[String,Int] withDefaultValue 0

    for (instanceDoc <- chosenone) {
      if (instanceDoc.author.toString.equals("Some(j_live)")){
        for (i <- instanceDoc.lyrics.split(" "))
          countsForJlive(i) = countsForJlive(i) + 1
      }
      else if (instanceDoc.author.toString.equals("Some(rakim)")){
        for(i <- instanceDoc.lyrics.split(" "))
          countsForrakim(i) += 1
      }
      else{
        for(i <- instanceDoc.lyrics.split(" "))
          countsForroots(i) += 1
      }


    }

    //println(countsForroots("[BAR]"))

   // conditional probability
    def conditionalProbability(inputString: String, authodName: String) : Double = {
      val alpha = 0.5
      if(authodName.equals("Jlive")){
        return 1.0 * ( countsForJlive(inputString) + alpha) / (countcForJlive.length + alpha*listOfAll.size)
      }

      else if(authodName.equals("rakim"))
        {
          return 1.0 * ( countsForrakim(inputString) + alpha) / (countcForrakim.length + alpha*listOfAll.size)
        }

     else
        return 1.0 * ( countsForroots(inputString) + alpha) / (countcForroots.length + alpha*listOfAll.size)
   }

    case class NGramLM(train: Instance, order: Int) extends CountLM {
      val vocab = train.lyrics.split(" ").toIndexedSeq.toSet
      val counts = new mutable.HashMap[List[String], Double] withDefaultValue 0.0
      val norm = new mutable.HashMap[List[String], Double] withDefaultValue 0.0
      for (i <- order until train.lyrics.split(" ").toIndexedSeq.length) {
        val history = train.lyrics.split(" ").toIndexedSeq.slice(i - order + 1, i).toList
        val word = train.author.get
        counts(word :: history) += 1.0
        norm(history) += 1.0
      }
    }

   // TODO given an instance, how would you classify it
    def classify(instance: Instance) : Option[String] ={

/*
     //unigram LM
     NGramLM(instance,1)

*/

     var resultForJlive = BigDecimal(periodOfJlive)
     var resultForrakim = BigDecimal(periodOfRakim)
     var resultForroots = BigDecimal(periodOfroots)

     var cau = List[String]()
     cau = instance.lyrics.split(" ").toList
     for(i <- cau){
       resultForJlive *= conditionalProbability(i,"Jlive")
       resultForrakim *= conditionalProbability(i,"rakim")
       resultForroots *= conditionalProbability(i,"roots")
     }
      //println(resultForJlive,resultForrakim,resultForroots)

     if(resultForJlive >= resultForrakim && resultForJlive >= resultForroots)
       return Some("j_live")
     else if(resultForrakim >= resultForJlive && resultForrakim >= resultForroots)
       return Some("rakim")
     else
       return Some("roots")

   }

    // execute your classifier  Seq[Instance]
    val predictions = dev.map(i => Instance(i.lyrics, i.author, classify(i)))
/*    for(i <- predictions){
      println(i.author,i.prediction)
      println("-------------")
    }*/
    // accurately predicted instances
    val accuratelyPredicted = predictions.map(i => i.author.get == i.prediction.get).count(_ == true)

    // total number of instances
    val totalInstances = predictions.length

    // evaluate accuracy
    val accuracy = 1.0 * accuratelyPredicted / totalInstances

    println("classification accuracy:" + accuracy)

  }



}
