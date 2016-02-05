package uk.ac.ucl.cs.mr.statnlpbook.assignment1

import java.io.File

import uk.ac.ucl.cs.mr.statnlpbook.Tokenizer
import uk.ac.ucl.cs.mr.statnlpbook.chapter.languagemodels.Util

/**
 * This stub is provided as help for you to solve the first question.
 * @author riedel
 */
object TokenizationQuestion {

  def main(args: Array[String]) {

    //the directory that contains the raw documents
    val dataDir = new File("/Users/HUANGWEIJIE/Desktop/scala/stat-nlp-book/data")

    //the file that contains the target gold tokenization
    val goldFile = new File("/Users/HUANGWEIJIE/Desktop/scala/stat-nlp-book/data/assignment1/p1/goldtokens.txt")

    //the actual raw documents
    val raw = Assignment1Util.loadTokenizationQuestionDocs(dataDir)

    //TODO: The tokenizer you need to improve to match the gold tokenization
    //Detailed rules
    // "[  ]" [BAR] [/BAR] [VERSE 1,2,3...]
    val BAR = "[\\]\\[]"
    val ruleForBAR = s"(?<!BAR)(?=$BAR)"
    val ruleForBAR2 = s"(?<=$BAR)(?!BAR)(?!\\/)"

    // " ' "  n't
    val quotationMark = "\\'"
    val ruleForQM = s"(?<!n)(?=$quotationMark)|(?=n\\'t)"

    // Dr.Mr.
    val fullstop = "[\\.]"
    val ruleforDr = s"(?<!Dr|Mr)(?=$fullstop)"
    val ruleforDr2 = s"(?<=$fullstop)"

    // Punctuations
    val symbol = "[\\,\"\\?\\!\\(\\)\\*\\{\\}\\;\\:\\+\\_]"
    val RuleForPunctuations = s"(?=$symbol)"
    val RuleForPunctuations2 = s"(?<=$symbol)"

    //Final Procession
    val tokenizer = Tokenizer.fromRegEx(s"(\\s|$ruleForBAR|$ruleForBAR2|$ruleForQM" +
      s"|$RuleForPunctuations|$RuleForPunctuations2|$ruleforDr|$ruleforDr2)")


    //the result of the tokenization
    val result = Util.words(raw map tokenizer)

    //the gold tokenization file
    val gold = Assignment1Util.loadWords(goldFile).toBuffer

    //Comparison between documents
    //println(raw.map(_.toText).mkString(" ").take(350))//Orginal one
    //println("---------")
    //println(result.mkString(" ").take(1750))//fixed one
    //println("---------")
    //println(gold.mkString(" ").take(1750))//correct one
    println("---------")

    //we find the first pair of tokens that don't match
    val mismatch = result.zip(gold).find { case (a, b) => a != b }

    //Your goal is to make sure that mismatch is None
    mismatch match {
      case None => println("Success!")
      case Some(pair) => println("The following tokens still don't match: " + pair)
    }

  }

}
