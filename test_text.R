#loading the pakages 
library(devtools)
install_github("bmschmidt/wordVectors")
library(wordVectors)
library(stringi)
library(data.table)
library(magrittr)
library(tidyverse)
library(plyr)
#download cookbook text files 
download.file("http://archive.lib.msu.edu/dinfo/feedingamerica/cookbook_text.zip","cookbooks.zip")
unzip("cookbooks.zip", exdir = "cookbooks")

prep_word2vec("cookbooks", "cookbooks.txt", lowercase = T)

#Training word2vec model
word.model =train_word2vec("cookbooks.txt", output="cookbook_vectors.bin",
                           threads=3, vectors = 100, window=12, force= T)

nearest_to(word.model, word.model[["fish"]])

#Example discharge note
example = "Adenocarcinoma of stomach with peritoneal carcinomatosis and massive ascite, stage IV under bidirection chemotherapy (neoadjuvant intraperitoneal-systemic chemotherapy) with intraperitoneal paclitaxel 120mg (20151126, 20151201) and systemic with Oxalip (20151127) and oral XELOX."

#Text process
text = tolower(example)
text = gsub("\n", "@@@@@", text, fixed = TRUE)
text = gsub("\r", "@@@@@", text, fixed = TRUE)
text = gsub("[ :,;-]", "@", text)
text = gsub("(", "@", text, fixed = TRUE)
text = gsub(")", "@", text, fixed = TRUE)
text = gsub("/", "@", text, fixed = TRUE)
text = strsplit(text, split = ".", fixed = TRUE)[[1]]
text = paste(text, collapse = "@@@@@")
text = strsplit(text, split = "@", fixed = TRUE)[[1]]

#Show result
text

word.data<-fread("wikipedia word2vec.txt", header = F, showProgress = F)
word.ref<-word.data%>%select(V1)%>%setDF%>%.[,1]%>%as.character
word.matrix<-word.data%>%select(-V1)%>%setDF%>%as.matrix

#aligning 
text.array<-matrix(0,nrow=length(text),ncol=50)
for(i in 1:length(text)){
  if(text[i]!=""){
    pos<-which(word.ref==text[i])
    if(length(pos)==1){
      text.array[i,]<-word.matrix[pos,]
    }
  }
}

library(imager)
image<-text.array
image[image>2]=2
image[image<-2]=-2
plot(as.cimg(t(image)))

#XNN training 
load("ICD10.RData")
Train.X.array = ARRAY[,,1:3000]
dim(Train.X.array) = c(100, 50, 1, 3000)
Train.Y = LABEL[1:3000]

Test.X.array = ARRAY[,,3001:5000]
dim(Test.X.array) = c(100, 50, 1, 2000)
Test.Y = LABEL[3001:5000]



cran <- getOption("repos")
cran["dmlc"] <- "https://s3-us-west-2.amazonaws.com/apache-mxnet/R/CRAN/"
options(repos = cran)
install.packages("mxnet")
library(mxnet)

library(mxnet)

get_symbol_textcnn <- function(num_outcome = 1, 
                               filter_sizes = 1:5, 
                               num_filter = c(40, 30, 15, 10, 5), 
                               Seq.length = 100, 
                               word.dimation = 50, 
                               dropout = 0.5) {
  
  data <- mx.symbol.Variable('data')
  
  concat_lst <- NULL
  
  for (i in 1:length(filter_sizes)) {
    convi <- mx.symbol.Convolution(data = data, 
                                   kernel = c(filter_sizes[i], word.dimation), 
                                   pad = c(filter_sizes[i]-1, 0), 
                                   num_filter = num_filter[i], 
                                   name = paste0('conv', i))
    relui <- mx.symbol.Activation(data = convi, 
                                  act_type = "relu", 
                                  name = paste0('relu', i))
    pooli <- mx.symbol.Pooling(data = relui, 
                               pool_type = "max", 
                               kernel = c(Seq.length + filter_sizes[i] - 1, 1), 
                               stride = c(1, 1), 
                               name = paste0('pool', i))
    concat_lst = append(concat_lst, pooli)
  }
  
  concat_lst$num.args = length(filter_sizes)
  
  h_pool = mxnet:::mx.varg.symbol.Concat(concat_lst)
  
  # dropout layer
  
  if (dropout > 0) {
    h_drop = mx.symbol.Dropout(data = h_pool, p = dropout)
  } else {
    h_drop = h_pool
  }
  
  # fully connected layer
  
  cls_weight = mx.symbol.Variable('cls_weight')
  cls_bias = mx.symbol.Variable('cls_bias')
  
  fc = mx.symbol.FullyConnected(data = h_drop, weight = cls_weight, bias = cls_bias, num_hidden = num_outcome)
  lr = mx.symbol.LogisticRegressionOutput(fc, name='lr')
  
  return(lr)
}

#Evaluation criteria
my.eval.metric.CE <- mx.metric.custom(
  name = "Cross-Entropy (CE)", 
  function(real, pred) {
    real1 = as.numeric(real)
    pred1 = as.numeric(pred)
    pred1[pred1 <= 1e-6] = 1e-6
    pred1[pred1 >= 1 - 1e-6] = 1 - 1e-6
    return(-mean(real1 * log(pred1) + (1 - real1) * log(1 - pred1), na.rm = TRUE))
  }
)

#Training a Model 
n.cpu <- 4
device.cpu <- lapply(0:(n.cpu-1), function(i) {mx.cpu(i)})

mx.set.seed(0)

cnn.model = mx.model.FeedForward.create(get_symbol_textcnn(),
                                        X = Train.X.array, y = Train.Y,
                                        ctx = device.cpu, num.round = 20,
                                        array.batch.size = 100, learning.rate = 0.05,
                                        momentum = 0.9, wd = 0.00001,
                                        eval.metric = my.eval.metric.CE)
#Predict the test data 
install.packages("pROC")
library(pROC)
pred.y = predict(cnn.model, Test.X.array)
ROC.test = roc(response = Test.Y, predictor = as.numeric(pred.y))
print(auc(ROC.test))
plot(ROC.test)
