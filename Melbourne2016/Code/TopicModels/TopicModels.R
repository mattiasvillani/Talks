
#install.packages("tm")
# install.packages("corpus.JSS.papers", repos = "http://datacube.wu.ac.at/", type = "source")
# install.packages("topicmodels")
library(tm)
library(corpus.JSS.papers)


# READING DATA AND PRE-PROCESS AWAY "WEIRD" ARTICLES - DON'T WORRY IF YOU CAN'T FOLLOW THIS PART

# Reading text from articles in Journal of Statistical Software using a special reader function.
data("JSS_papers", package = "corpus.JSS.papers")

# Extract only papers up to 2010-08-05 and remove papers with weird (non-ASCII characters) in abstract
# Note the Encoding function returns "unknown" for ASCII text.
JSS_papers <- JSS_papers[JSS_papers[,"date"] < "2010-08-05",]
JSS_papers <- JSS_papers[sapply(JSS_papers[, "description"], Encoding) == "unknown",]

# Removing HTML markup for subscripting and greek letters etc for reading to corpus
#install.packages("XML")
library("XML")
remove_HTML_markup <- function(s) tryCatch({ doc <- htmlTreeParse(paste("<!DOCTYPE html>", s), asText = TRUE, trim = FALSE)
                                             xmlValue(xmlRoot(doc))}, error = function(s) s)
corpus <- Corpus(VectorSource(sapply(JSS_papers[, "description"], remove_HTML_markup)))


# Construct DocumentTerm matrix
# Using some linguistic pre-processing (remove stopwords and punctuation etc)
Sys.setlocale("LC_COLLATE", "C") # Language setting for the linguistic analysis
JSS_dtm <- DocumentTermMatrix(corpus, control = list(stopwords = TRUE, 
                             minWordLength = 3, removeNumbers = TRUE, removePunctuation = TRUE))
JSS_dtm <- DocumentTermMatrix(corpus)
dim(JSS_dtm)
inspect(JSS_dtm[1:20,'algorithm'])

# Reducing the number of features by keeping only words with tf-idf > 0.1
#install.packages("slam")
library("slam")
term_tfidf <- tapply(JSS_dtm$v/row_sums(JSS_dtm)[JSS_dtm$i], JSS_dtm$j, mean) * log2(nDocs(JSS_dtm)/col_sums(JSS_dtm > 0))
JSS_dtm <- JSS_dtm[,term_tfidf >= 0.1] # Removing words with low tf-idf
JSS_dtm <- JSS_dtm[row_sums(JSS_dtm) > 0,] # Removing documents where no feature is present.

# Fitting a topic model with 30 topics
library(topicmodels)
LDAfit <- LDA(JSS_dtm[1:340,], k = 10, control = list(seed = 2010, iter=2000, keep=1, alpha=0.01, delta=0.01), method = "Gibbs") 

# Check convergence
plot(LDAfit@logLiks, type="l")

# Look at the results
mostLikelyTopics <- topics(LDAfit, 2)
mostLikelyWords <- terms(LDAfit, 5)

# Finding the most probable topic in each issue in volume 24
topics_v24 <- topics(LDAfit)[grep("/v24/", JSS_papers[, "identifier"])]
most_frequent_v24 <- which.max(tabulate(topics_v24))
terms(LDAfit, 10)[, most_frequent_v24]

# Predicting the last four document in the corpus (which was left out in the estimation/training)
postNewData <- posterior(LDAfit, newdata = JSS_dtm[341:344,])
perplexity(object = LDAfit, JSS_dtm[341:344,])
LDAfit20 <- LDA(JSS_dtm[1:340,], k = 20, control = list(seed = 2010, iter=2000, keep=1, alpha=0.01, delta=0.01), method = "Gibbs") 
perplexity(object = LDAfit20, JSS_dtm[341:344,])
# Though it is just three hold out documents... and I'm not sure how this is calculated...


postNewData$topics
terms(LDAfit, 10)[,c(1,8)] # Document 347 talks mainly about topic 1 and 8
inspect(corpus[347])

