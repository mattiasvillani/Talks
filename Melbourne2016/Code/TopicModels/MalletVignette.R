# On my mac, I need to issue the following command in terminal:
# sudo R CMD javareconf
# sudo ln -f -s $(/usr/libexec/java_home)/jre/lib/server/libjvm.dylib /usr/local/lib

install.packages("tm")     # Text mining package
install.packages("mallet") # Binders to Java toolkit Mallet
install.packages("XML")
library("tm")
library("mallet")
library(XML)

# Loading Reuters data
reut21578 <- system.file("texts", "crude", package = "tm")
reuters <- VCorpus(DirSource(reut21578), readerControl = list(reader = readReut21578XMLasPlain))
reuters_text_vector <- unlist(lapply(reuters, as.character))

# List of stop words
stopwords_en <- system.file("stopwords/english.dat", package = "tm")

# Setting up a Mallet instance
mallet.instances <- mallet.import(id.array = as.character(1:length(reuters_text_vector)), 
                                  text.array = reuters_text_vector, 
                                  stoplist.file = stopwords_en,
                                  token.regexp = "\\p{L}[\\p{L}\\p{P}]+\\p{L}")

# Create a topic trainer instance
topic.model <- MalletLDA(num.topics=5, alpha.sum = 1, beta = 0.1)

# Load our documents. We could also pass in the filename of a saved instance list file that we build from the command-line tools.
topic.model$loadDocuments(mallet.instances)

# Get the vocabulary, and some statistics about word frequencies. These may be useful in further curating the stopword list.

vocabulary <- topic.model$getVocabulary()
head(vocabulary)

word.freqs <- mallet.word.freqs(topic.model)
head(word.freqs)

# Optimize hyperparameters every 20 iterations, after 50 burn-in iterations.
topic.model$setAlphaOptimization(20, 50)

# Train/estimate the model 
topic.model$train(200)

# Optimize rather than sample
# topic.model$maximize(10)

# Get the probability of topics in documents and the probability of words in topics. 
# By default, these functions return raw word counts. Here we want probabilities,so we normalize, 
# and add "smoothing" so that nothing has exactly 0 probability.
doc.topics <- mallet.doc.topics(topic.model, smoothed=TRUE, normalized=TRUE)
topic.words <- mallet.topic.words(topic.model, smoothed=TRUE, normalized=TRUE)

# What are the top words in topic 2? 
# Notice that R indexes from 1 and Java from 0, so this will be the topic that mallet called topic 1.
mallet.top.words(topic.model, word.weights = topic.words[2,], num.top.words = 5)

#Show the first document with at least 5% tokens belonging to topic 1.
inspect(reuters[doc.topics[,1] > 0.05][1])

#How do topics differ across different sub-corpora?
usa_articles <- unlist(meta(reuters, "places")) == "usa"
usa.topic.words <- mallet.subset.topic.words(topic.model, 
                                             subset.docs = usa_articles,
                                             smoothed=TRUE, 
                                             normalized=TRUE)
other.topic.words <- mallet.subset.topic.words(topic.model, 
                                               subset.docs = !usa_articles,
                                               smoothed=TRUE, 
                                               normalized=TRUE)
head(mallet.top.words(topic.model, usa.topic.words[1,]))
head(mallet.top.words(topic.model, other.topic.words[1,]))
