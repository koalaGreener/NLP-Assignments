package uk.ac.ucl.cs.mr.statnlpbook.assignment2

import scala.collection.mutable

/**
 * Created by Georgios on 11/11/2015.
 */
object Problem2 {


  /**
   * Train a linear model using the average perceptron algorithm.
   * @param instances the training instances.
   * @param feat a joint feature function.
   * @param predict a prediction function that maps inputs to outputs using the given weights.
   * @param iterations number of iterations.
   * @param learningRate
   * @tparam X type of input.
   * @tparam Y type of output.
   * @return a linear model trained using the perceptron algorithm.
   */
  def trainAvgPerceptron[X, Y](instances: Seq[(X, Y)],
                               feat: (X, Y) => FeatureVector,
                               predict: (X, Weights) => Y,
                               iterations: Int = 2,
                               learningRate:Double = 1.0): Weights = {

    val tempWeights = new mutable.HashMap[FeatureKey, Double]() withDefaultValue 0.0
    val returnWeights = new mutable.HashMap[FeatureKey, Double]() withDefaultValue 0.0
    var times = 0

    for( a <- 1 to iterations){
      for(instance <- instances){
        val prediectedLabel = predict(instance._1, tempWeights)
        if(instance._2 != prediectedLabel) {
          addInPlace(feat(instance._1, instance._2), tempWeights, learningRate)
          addInPlace(feat(instance._1, prediectedLabel), tempWeights, -learningRate)
        }
        for ((k, v) <- tempWeights){
          if(tempWeights(k) != 0){
            val add = returnWeights(k)
            returnWeights.update(k, v + add)
          }
        }
      }
    }

    //returnWeights.map { case (k, v) => (k, v/times) }

    return returnWeights
  }

  /**
   * Run this code to evaluate your implementation of your avereaged perceptron algorithm trainer
   * Results should be similar to the precompiled trainer
   * @param args
   */
  def main (args: Array[String] ) {

    //times
    val starttime = System.currentTimeMillis

    val train_dir = "./data/assignment2/bionlp/train"

    // load train and dev data
    // read the specification of the method to load more/less data for debugging speedup
    val (trainDocs, devDocs) = BioNLP.getTrainDevDocuments(train_dir, 0.8, 100)
    // make tuples (Candidate,Gold)
    def preprocess(candidates: Seq[Candidate]) = candidates.map(e => e -> e.gold)

    // ================= Trigger Classification =================

    // get candidates and make tuples with gold
    // read the specifications of the method for different subsampling thresholds
    // no subsampling for dev/test!
    def getTriggerCandidates(docs: Seq[Document]) = docs.flatMap(_.triggerCandidates(0.02))
    def getTestTriggerCandidates(docs: Seq[Document]) = docs.flatMap(_.triggerCandidates())
    val triggerTrain = preprocess(getTriggerCandidates(trainDocs))
    val triggerDev = preprocess(getTestTriggerCandidates(devDocs))

    // get label set
    val triggerLabels = triggerTrain.map(_._2).toSet

    // define model
    val triggerModel = SimpleClassifier(triggerLabels, defaultTriggerFeatures)

    val myWeights = trainAvgPerceptron(triggerTrain, triggerModel.feat, triggerModel.predict, 1)
    val precompiledWeights = PrecompiledTrainers.trainAvgPerceptron(triggerTrain, triggerModel.feat, triggerModel.predict, 1)

    // get predictions on dev
    val (myPred, gold) = triggerDev.map { case (trigger, gold) => (triggerModel.predict(trigger, myWeights), gold) }.unzip
    val (precompiledPred, _) = triggerDev.map { case (trigger, gold) => (triggerModel.predict(trigger, precompiledWeights), gold) }.unzip

    // evaluate models (dev)
    println("Evaluation - my trainer:")
    println(Evaluation(gold, myPred, Set("None")).toString)
    println("Evaluation - precompiled trainer:")
    println(Evaluation(gold, precompiledPred, Set("None")).toString)
    println(System.currentTimeMillis - starttime)

  }

  def defaultTriggerFeatures(x: Candidate, y: Label): FeatureVector = {
    val doc = x.doc
    val begin = x.begin
    val end = x.end
    val thisSentence = doc.sentences(x.sentenceIndex) //use this to gain access to the parent sentence
    val feats = new mutable.HashMap[FeatureKey,Double]
    feats += FeatureKey("label bias", List(y)) -> 1.0 //bias feature
    val token = thisSentence.tokens(begin) //first token of Trigger
    feats += FeatureKey("first trigger word", List(token.word, y)) -> 1.0 //word feature
    feats.toMap
  }


}
