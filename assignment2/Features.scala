package uk.ac.ucl.cs.mr.statnlpbook.assignment2

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/**
 * Finished by Weijie HUANG on 19/12/2015.
 */

object Features {

  /**
   * a feature function with two templates w:word,label and l:label.
   * Example for Trigger Exraction
   * @param x
   * @param y
   * @return
   */
  def defaultTriggerFeatures(x: Candidate, y: Label): FeatureVector = {
    val doc = x.doc
    val begin = x.begin
    val end = x.end
    val thisSentence = doc.sentences(x.sentenceIndex) //use this to gain access to the parent sentence
    //-----------------------
    val feats = new mutable.HashMap[FeatureKey,Double]
    //List(y)应该是全部training set出现过的label都在这里了
    feats += FeatureKey("label bias", List(y)) -> 1.0 //bias feature
    //-----------------------
    val token = thisSentence.tokens(begin) //first token of Trigger
    feats += FeatureKey("first trigger word", List(token.word, y)) -> 1.0
    feats.toMap
  }
  /**
   * a feature function with two templates w:word,label and l:label.
   * Example for Argument Exraction
   * @param x
   * @param y
   * @return
   */
  // y = label set
  def defaultArgumentFeatures(x: Candidate, y: Label): FeatureVector = {
    val doc = x.doc
    val begin = x.begin
    val end = x.end
    val thisSentence = doc.sentences(x.sentenceIndex)
    //-----------------------
    val feats = new mutable.HashMap[FeatureKey,Double]
    feats += FeatureKey("label bias", List(y)) -> 1.0
    //-----------------------
    val token = thisSentence.tokens(begin) //first word of argument
    feats += FeatureKey("first argument word", List(token.word, y)) -> 1.0
    //-----------------------
    val event = thisSentence.events(x.parentIndex) //use this to gain access to the parent event
    val eventHeadToken = thisSentence.tokens(event.begin) //first token of event
    feats += FeatureKey("is protein_first trigger word", List(x.isProtein.toString,eventHeadToken.word, y)) -> 1.0
    feats.toMap
  }

  def myTriggerFeatures(x: Candidate, y: Label): FeatureVector = {
    val doc = x.doc
    val begin = x.begin
    val end = x.end
    val thisSentence = doc.sentences(x.sentenceIndex) //use this to gain access to the parent sentence
    val token = thisSentence.tokens(begin) //first token of Trigger

    def isContainedWithTheSpecialWords(x: ListBuffer[String]): Boolean = {
      var returnValue = false
      for(i <- 0 to x.size - 1  ){
        if (token.word.contains(x(i)))
          returnValue = true
      }
      return returnValue
    }

    //-----------------------
    val feats = new mutable.HashMap[FeatureKey,Double]
    //List(y)应该是全部training set出现过的label都在这里了
    feats += FeatureKey("label_bias_Trigger", List(y)) -> 1.0 //bias feature
    //-----------------------
    feats += FeatureKey("first_trigger_word_Trigger", List(token.word, y)) -> 1.0
    //-----------------------
    if(thisSentence.mentions.length == 0) //不存在mentions的情况 直接判断length为0
      feats += FeatureKey("first_mention_word_Trigger", List(y)) -> 1.0
    else{
      for(i <- 0 to 1) //用循环少写一点代码 mentions 0 29.76% | 0-1 36.31% | 0-2 30.04% | 0-20 26.35%
      {
        if(thisSentence.mentions.isDefinedAt(i))
          feats += FeatureKey("first_mention_word_Trigger", List(thisSentence.tokens(thisSentence.mentions(i).begin).word, y)) -> 1.0
      }
    }
    //-----------------------
    //deps内关系和一对构成关系的词和label组成argument
    val deps = thisSentence.deps
    for(i <- 0 to 120){
      if(deps.isDefinedAt(i))
        feats += FeatureKey("Dep_Trigger", List(deps(i).label, thisSentence.tokens(deps(i).head).word, thisSentence.tokens(deps(i).mod).word,y)) -> 1.0
    }
    //-----------------------
    //5个词作为窗口 分别抓2个和3个构成bigram和trigram
    //Trigram窗口选4 Bigram窗口选5
    var startIndexOfTokenFortrigram = begin - 4
    var endIndexOfTokenFortrigram = begin + 4

    if (startIndexOfTokenFortrigram >= 0 && endIndexOfTokenFortrigram <= thisSentence.tokens.length - 1 ){
      for (i <- startIndexOfTokenFortrigram to (endIndexOfTokenFortrigram -2) )
      {
        feats += FeatureKey("trigram_Trigger", List(thisSentence.tokens(startIndexOfTokenFortrigram).word,thisSentence.tokens(startIndexOfTokenFortrigram + 1).word, thisSentence.tokens(startIndexOfTokenFortrigram + 2).word,y)) -> 1.0 //trigram
      }
    }
    //-----------------------
    //Bigram
    var startIndexOfTokenForBigram = begin - 5
    var endIndexOfTokenForBigram = begin + 5

    if (startIndexOfTokenForBigram >= 0 && endIndexOfTokenForBigram <= thisSentence.tokens.length - 1 ) {
      for (i <- startIndexOfTokenForBigram to (endIndexOfTokenForBigram -1) )
      {
        feats += FeatureKey("bigram_Trigger", List(thisSentence.tokens(startIndexOfTokenForBigram).word,thisSentence.tokens(startIndexOfTokenForBigram + 1).word,y)) -> 1.0 //bigram
      }
    }
    //-----------------------
    //紧邻前后的单词和Trigger构成一个argument
    if(begin - 1 >= 0)
      feats += FeatureKey("BeforeAndAfter_Trigger", List(thisSentence.tokens(begin-1).word,thisSentence.tokens(begin).word, thisSentence.tokens(begin+1).word,y)) -> 1.0 //BeforeAndAfter
    //-----------------------
    //3个单词以内是否有蛋白质 boolean,蛋白质名称,y作为argument
    for (i <- 0 to 200){
      if(thisSentence.mentions.isDefinedAt(i)) {
        val indexOfProteinStart = thisSentence.mentions(i).begin
        val indexOfProteinEnd = thisSentence.mentions(i).end
        if ((begin >= indexOfProteinStart && begin <= indexOfProteinEnd)||(begin >= indexOfProteinEnd && begin <= indexOfProteinStart) ){
          feats += FeatureKey("proteinExistInThreeTokens_Trigger", List("true",thisSentence.tokens(thisSentence.mentions(i).begin).word,y)) -> 1.0 //proteinExistInThreeTokens
          //only begin 2923 only end 2681 feats += FeatureKey("proteinExistInThreeTokens", List("true",thisSentence.tokens(thisSentence.mentions(i).end).word,y)) -> 1.0 //proteinExistInThreeTokens
        }
      }
    }
    //-----------------------
    // The first trigger word that appears with Protein_catabolism
    var storageForProtein_catabolism = mutable.ListBuffer[String]("degrade")
    feats += FeatureKey("Protein_catabolism_Trigger", List(isContainedWithTheSpecialWords(storageForProtein_catabolism).toString,y)) -> 1.0


    // The first trigger word that appears with Phosphorylation
    var storageForPhosphorylation = mutable.ListBuffer[String]("Phosphorylated")
    feats += FeatureKey("Phosphorylation_Trigger", List(isContainedWithTheSpecialWords(storageForPhosphorylation).toString, y)) -> 1.0


    // The first trigger word that appears with Transcription
    var storageForTranscription = mutable.ListBuffer[String]( "expression", "transcripts")
    feats += FeatureKey("Transcription_Trigger", List(isContainedWithTheSpecialWords(storageForTranscription).toString, y)) -> 1.0


    // The first trigger word that appears with Localization
    var storageForLocalization = mutable.ListBuffer[String]("localize","secretion", "translocation","presence","release","abundance", "migrating","mobilized")
        feats += FeatureKey("Localization_Trigger", List(isContainedWithTheSpecialWords(storageForLocalization).toString, y)) -> 1.0


    // The first trigger word that appears with Binding
    var storageForBinding = mutable.ListBuffer[String]("bind","interaction","ligation" ,"bound", "associates","interacts","cross-linking","associated",  "participation", "coimmunoprecipitated", "pair" )

        feats += FeatureKey("Binding_Trigger", List(isContainedWithTheSpecialWords(storageForBinding).toString, y)) -> 1.0


    // The first trigger word that appears with  Regulation
    var storageForRegulation = mutable.ListBuffer[String]("NF-kappaB-dependent", "Rel/NF-kappaB-responsive" , "cycloheximide-sensitive", "CsA-sensitive", "role", "effect", "regulates", "affecting","dysregulation","modulating","target","correlation","modulate")
        feats += FeatureKey("Regulation_Trigger", List(isContainedWithTheSpecialWords(storageForRegulation).toString,  y)) -> 1.0


    // The first trigger word that appears with Negetive Regulation
    var storageForNRegulation = mutable.ListBuffer[String]("abolished","reduction","limited")

        feats += FeatureKey("N-Regulation_Trigger", List(isContainedWithTheSpecialWords(storageForNRegulation).toString, y)) -> 1.0


    // The first trigger word that appears with Positive Regulation
    var storageForAllRegulation = mutable.ListBuffer[String]("increasing", "accumulation","enhanced", "increased","upregulated", "elevated","increase", "synergize", "accelerates","induction", "induced",  "promote")
        feats += FeatureKey("P-Regulation_Trigger", List(isContainedWithTheSpecialWords(storageForAllRegulation).toString,  y)) -> 1.0


    // The first trigger word that appears with None
    var storageForNone = mutable.ListBuffer[String]("for", "For", "of", "at", "amounts","with","to","by","that","are"," To",":","from","requirement","showing","Exclusion","presented","analyzed")
    feats += FeatureKey("None_Trigger", List(isContainedWithTheSpecialWords(storageForNone).toString, y)) -> 1.0

    //-----------------------
    //Positive / Negative Regulation trigger is Verb and realtion bewteen trigger and protein is  dobj
    //thisSentence.mentions.map(_.begin)
    var yesorno = false
    for (i <- 0 to 10)
      if (thisSentence.mentions.isDefinedAt(i))
        {
          if (thisSentence.deps.map(_.head).contains(thisSentence.mentions(i).begin))
            yesorno = true
        }


    feats += FeatureKey("Dobj_Trigger", List(token.word, thisSentence.tokens(begin).pos.equals("VB").toString,thisSentence.deps.map(_.label).equals("dobj").toString,thisSentence.deps.map(_.mod).contains(begin).toString, yesorno.toString ,y)) -> 1.0

    //last trigger word is prep and relation bewteen trigger word and protein is agent
    feats += FeatureKey("prep_Trigger", List(token.word,thisSentence.tokens(end).pos.equals("prep").toString,thisSentence.deps.map(_.label).equals("agent").toString,thisSentence.deps.map(_.mod).contains(begin).toString, yesorno.toString ,y)) -> 1.0

    //dep between two words
    feats += FeatureKey("dep_Trigger", List(token.word,thisSentence.deps.map(_.label).equals("dep").toString,thisSentence.deps.map(_.mod).contains(begin).toString, yesorno.toString, y)) -> 1.0

    //Trigger exists bewteen first protein and another protein
    if(thisSentence.mentions.isDefinedAt(1)) {
      feats += FeatureKey("Mid_Trigger", List(token.word,(thisSentence.mentions(0).begin < begin).toString, (thisSentence.mentions(1).end > begin).toString, y)) -> 1.0
    }

    //Trigger word是否大写 = 不等于自身转化为小写
    feats += FeatureKey("Contains capital letter", List(token.word, token.word.equals(token.word.toLowerCase).toString, y)) -> 1.0

    feats.toMap
  }



  def myArgumentFeatures(x: Candidate, y: Label): FeatureVector = {
    val doc = x.doc
    val begin = x.begin
    val end = x.end
    val thisSentence = doc.sentences(x.sentenceIndex)
    val event = thisSentence.events(x.parentIndex) //use this to gain access to the parent event
    val eventHeadToken = thisSentence.tokens(event.begin) //first token of event

    def isContainedWithTheSpecialWords(x: ListBuffer[String]): Boolean = {
      var returnValue = false
      for(i <- 0 to x.size - 1  ){
        if (eventHeadToken.word.equals(x(i)))
          returnValue = true
      }
      return returnValue
    }


    //-----------------------
    val feats = new mutable.HashMap[FeatureKey,Double]
    feats += FeatureKey("label bias_Argument", List(y)) -> 1.0
    //-----------------------
    val token = thisSentence.tokens(begin) //first word of argument
    val tokenEnd = thisSentence.tokens(end) //first word of argument
    feats += FeatureKey("first argument word_Argument", List(token.word, y)) -> 1.0

    //-----------------------
    feats += FeatureKey("is protein_first trigger word_Argument", List(x.isProtein.toString,eventHeadToken.word, y)) -> 1.0
    //-----------------------
    //前后events所对应的first trigger word
    if (thisSentence.events.isDefinedAt(x.parentIndex - 1 ) && thisSentence.events.isDefinedAt(x.parentIndex + 1 )){
      val eventBefore = thisSentence.events(x.parentIndex - 1)
      val eventBeforeHeadToken = thisSentence.tokens(eventBefore.begin) //first token of event
      val eventAfter = thisSentence.events(x.parentIndex + 1)
      val eventAfterHeadToken = thisSentence.tokens(eventAfter.begin) //first token of event
      feats += FeatureKey("3 nearby events's first trigger word_Argument", List(eventBeforeHeadToken.word,eventHeadToken.word,eventAfterHeadToken.word, y)) -> 1.0
    }
    //-----------------------
    //deps 对应words之间的关系
    for (i <- 0 to thisSentence.deps.length -1 )
      {
        val indexHead = thisSentence.deps(i).head
        val indexMod = thisSentence.deps(i).mod

        feats += FeatureKey("3 deps content_Argument", List(thisSentence.tokens(indexHead).word,thisSentence.tokens(indexMod).word,thisSentence.deps(i).label, y)) -> 1.0
      }
    //deps中对应的父节点
    //pos中对应的动词或其他等


    //-----------------------
    //trigram
    var startIndexOfTokenFortrigram = thisSentence.events(x.parentIndex).begin - 2
    var endIndexOfTokenFortrigram = thisSentence.events(x.parentIndex).begin + 2

    if (startIndexOfTokenFortrigram >= 0 && endIndexOfTokenFortrigram <= thisSentence.tokens.length - 1 ){
      for (i <- startIndexOfTokenFortrigram to (endIndexOfTokenFortrigram -2) )
      {
        feats += FeatureKey("trigram_Argument", List(thisSentence.tokens(startIndexOfTokenFortrigram).word,thisSentence.tokens(startIndexOfTokenFortrigram + 1).word, thisSentence.tokens(startIndexOfTokenFortrigram + 2).word,y)) -> 1.0 //trigram
      }
    }
    //-----------------------
    //Bigram
    var startIndexOfTokenForBigram = thisSentence.events(x.parentIndex).begin - 2
    var endIndexOfTokenForBigram = thisSentence.events(x.parentIndex).begin + 2

    if (startIndexOfTokenForBigram >= 0 && endIndexOfTokenForBigram <= thisSentence.tokens.length - 1 ) {
      for (i <- startIndexOfTokenForBigram to (endIndexOfTokenForBigram -1) )
      {
        feats += FeatureKey("bigram_Argument", List(thisSentence.tokens(startIndexOfTokenForBigram).word,thisSentence.tokens(startIndexOfTokenForBigram + 1).word,y)) -> 1.0 //bigram
      }
    }
    //-----------------------
    // TAG link with argument 词性 Base form
    feats += FeatureKey("posTag_Argument", List(thisSentence.tokens(begin).pos, y)) -> 1.0

    //-----------------------
    // Dep's label with mention's first word
    for (i <- 0 to 20) {
      if (thisSentence.mentions.isDefinedAt(i) && thisSentence.deps.isDefinedAt(i)) {
        feats += FeatureKey("Deps's label_First mention word_Argument", List(thisSentence.deps(i).label, thisSentence.tokens(thisSentence.mentions(i).begin).word, y)) -> 1.0
      }
    }
    //-----------------------
    // The first trigger word that indicates the Regulation that comes with Cause
    var storageForAllRegulation = mutable.ListBuffer[String]("increasing", "accumulation","enhanced", "increased","upregulated", "elevated","increase", "synergize", "accelerates","induction", "induced",  "promote","NF-kappaB-dependent", "Rel/NF-kappaB-responsive" , "cycloheximide-sensitive", "CsA-sensitive", "role", "effect", "regulates", "affecting","modulating","target","correlation","modulate","abolished","reduction","limited")
    feats += FeatureKey("Frequent_word_Regulation_Cause_Argument", List(isContainedWithTheSpecialWords(storageForAllRegulation).toString, eventHeadToken.word, y)) -> 1.0

/*    PREP TO RULE
    DEPENDENCY= prep to(Trigger,Participant) AND
      TRIGGER POS= VB∗ AND
    TRIGGER in (lead, contribute) OR EVENT TYPE= Binding
    ⇒ PARTICIPANT ROLE= Theme*/

    var Prep_to_Rule_boolean = false
    for (dep <- thisSentence.deps)
    {
      for (i <- 0 to thisSentence.mentions.size - 1 ){
        if (dep.head == (thisSentence.mentions(i).begin))
          if (dep.label.equals("prep_to"))
            if(dep.mod == event.begin)
              if(eventHeadToken.pos.equals("VB"))
                Prep_to_Rule_boolean = true
      }
    }
    val Prep_to_Rule_Words = mutable.ListBuffer[String]("lead", "contribute")
    feats += FeatureKey("Prep_to_Rule_Argument", List(Prep_to_Rule_boolean.toString, isContainedWithTheSpecialWords(Prep_to_Rule_Words).toString, y)) -> 1.0

/*

/*    #Example: . . . interaction is crucial for v-erb function.
    NSUBJ RULE
      DEPENDENCY= nsubj(Trigger,Participant) AND
      TRIGGER POS= JJ∗ AND
    EVENT TYPE= ∗Regulation
      TRIGGER in (essential, critical, crucial, necessary, responsible, important, sufficient)
    ⇒ PARTICIPANT ROLE= Cause*/

    var NSUBJ_RULE_boolean = false
    for (dep <- thisSentence.deps)
    {
      for (i <- 0 to thisSentence.mentions.size - 1 ){
        if (dep.head == (thisSentence.mentions(i).begin))
          if (dep.label.equals("nsubj"))
            if(dep.mod == event.begin)
              if(eventHeadToken.pos.equals("JJ"))
                NSUBJ_RULE_boolean = true
      }
    }
    val NSUBJ_RULE_Words = mutable.ListBuffer[String]("essential", "critical", "crucial", "necessary", "responsible", "important", "sufficient")
    feats += FeatureKey("NSUBJ_RULE_Argument", List(NSUBJ_RULE_boolean.toString, isContainedWithTheSpecialWords(NSUBJ_RULE_Words).toString, y)) -> 1.0

/*    #Example: . . . interaction between P50-P65 and c-Jun complexes . . .
    PREP BETWEEN RULE
    DEPENDENCY= prep between(Trigger,Participant) AND
      TRIGGER POS= NN∗ AND
    TRIGGER in (association, interaction)
    ⇒ PARTICIPANT ROLE= Theme*/

    var PREP_BETWEEN_RULE_boolean = false
    for (dep <- thisSentence.deps)
    {
      for (i <- 0 to thisSentence.mentions.size - 1 ){
        if (dep.head == (thisSentence.mentions(i).begin))
          if (dep.label.equals("prep_between"))
            if(dep.mod == event.begin)
              if(eventHeadToken.pos.equals("NN"))
                PREP_BETWEEN_RULE_boolean = true
      }
    }
    val PREP_BETWEEN_RULE_words = mutable.ListBuffer[String]("association", "interaction")
    feats += FeatureKey("PREP_BETWEEN_RULE_Argument", List(PREP_BETWEEN_RULE_boolean.toString, isContainedWithTheSpecialWords(PREP_BETWEEN_RULE_words).toString, y)) -> 1.0

/*    #Example: . . . I kappa B alpha, which prevented translocation . . .
    RCMOD RULE
      DEPENDENCY= rcmod(Participant,Trigger) AND
      TRIGGER POS= VB∗ AND
    TRIGGER in (involve, require) OR EVENT TYPE= Binding
    ⇒ PARTICIPANT ROLE= Theme
    TRIGGER POS= NN∗ AND
    EVENT TYPE= ∗Regulation
    ⇒ PARTICIPANT ROLE= Cause*/

    var RCMOD_RULE_boolean = false
    for (dep <- thisSentence.deps)
    {
      for (i <- 0 to thisSentence.mentions.size - 1 ){
        if (dep.mod == (thisSentence.mentions(i).begin))
          if (dep.label.equals("rcmod"))
            if(dep.head == event.begin)
              if(eventHeadToken.pos.equals("VB"))
                RCMOD_RULE_boolean = true
      }
    }
    val RCMOD_RULE_words = mutable.ListBuffer[String]("involve", "require")
    feats += FeatureKey("RCMOD_RULE_Argument", List(RCMOD_RULE_boolean.toString, isContainedWithTheSpecialWords(RCMOD_RULE_words).toString, y)) -> 1.0

/*    #Example: . . . IL-4 activates . . . by inducing tyrosine phosphorylation . . .
    PREPC BY-NSUBJ RULE
      DEPENDENCY= prepc by(X,Trigger) AND nsubj(X,Participant) AND
      TRIGGER POS= VB∗ AND
    TRIGGER in (involve, require) OR EVENT TYPE= Binding
    ⇒ PARTICIPANT ROLE= Theme
    EVENT TYPE= ∗Regulation
    ⇒ PARTICIPANT ROLE= Cause*/

var PREPC_BY_NSUBJ_boolean = false

      if(eventHeadToken.pos.equals("VB"))
      if(thisSentence.deps.map(_.label).contains("prepc_by"))
        if(thisSentence.deps.map(_.label).contains("nsubj"))
            if(thisSentence.deps.map(_.head).contains(event.begin))
              for (i <- 0 to thisSentence.mentions.size - 1 ){
                if (thisSentence.deps.map(_.head).contains(thisSentence.mentions(i).begin))
                  PREPC_BY_NSUBJ_boolean = true
              }

    val PREPC_BY_NSUBJ_RULE_words = mutable.ListBuffer[String]("involve", "require")
    feats += FeatureKey("PREPC_BY_NSUBJ_RULE_Argument", List(PREPC_BY_NSUBJ_boolean.toString, isContainedWithTheSpecialWords(PREPC_BY_NSUBJ_RULE_words).toString, y)) -> 1.0


/*    #Example: The role of P16 in . . . thymidine kinase regulation . . . #Corrective rule
      PREP OF-PREP IN RULE
    DEPENDENCY= prep of(Trigger,X) AND prep in(X,Participant) AND
      TRIGGER POS= NN∗ AND
    TRIGGER in (role, importance, involvement, increase, decline, change, reduction, decrease)
    ⇒ PARTICIPANT ROLE= Theme*/

    var PREP_OF_PREP_IN_boolean = false

    if(eventHeadToken.pos.equals("NN"))
      if(thisSentence.deps.map(_.label).contains("prepc_of"))
        if(thisSentence.deps.map(_.label).contains("prep_in"))
          if(thisSentence.deps.map(_.mod).contains(event.begin))
            for (i <- 0 to thisSentence.mentions.size - 1 ){
              if (thisSentence.deps.map(_.head).contains(thisSentence.mentions(i).begin))
                PREP_OF_PREP_IN_boolean = true
            }

    val PREP_OF_PREP_IN_RULE_words = mutable.ListBuffer[String]("role", "importance", "involvement","increase","decline","change","reduction","decrease")
    feats += FeatureKey("PREP_OF_PREP_IN_RULE_Argument", List(PREP_OF_PREP_IN_boolean.toString, isContainedWithTheSpecialWords(PREP_OF_PREP_IN_RULE_words).toString, y)) -> 1.0
*/
    feats.toMap
}}
