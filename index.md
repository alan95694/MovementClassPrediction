# Practical Machine Learning Cousera Project
CW  
August 14, 2016  



## Algorithmic Exercise Quality Classification
# Executive Summary
A great deal of work has been done to quantify how many repetitions of an 
exercise have been done but little work has been done to algorithmically determine
exercise quality.  This report details a machine learning approach that classifies
if an exercise was done correctly or in what manner it was not done correctly.

# Introduction and Background
As stated in the problem specification:
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks.  In this report we will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to algorithmically predict if an exercise was performed correctly or not.  Each participant asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: [http://groupware.les.inf.puc-rio.br/har](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset).

Raw Training data was obtained from [https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

Raw Testing data was obtained from [https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

All analysis was performed in the R environment with the caret, parallel and doParallel libraires.



Raw data is loaded from a local file, data files contained summary rows keyed with 
new_window = "yes" these rows were removed to focus on a realtime analysis method. 
Also many columns only contained non-NA data when the row represented summary data
these columns were also removed.  Finally potentially misleading bookkeeping information
was removed from columns 1 to 7 this included subject name, time of day etc.  Without detailed
knowledge of the test set up it is possible that time of day is related to the exercise 
quality (if data was systematically collected in one manner) and this could incorrectly be 
used by the algorithm to predict the exercise quality.  Data was also split into 
predictors and matching responses.


```r
    fname <- "pml-training.csv" # data file is within current working directory
    data <- read.csv(file = fname, header = TRUE, sep = ",", stringsAsFactors = TRUE)

    ## Remove summary rows, keep only non-summary rows
    data <- data[ data$new_window == "no", ]
    ## Remove columns with only NA
    data <- data[ colSums(!is.na(data)) > 0]
    ## Remove empty columsn
    data <- data[ , data[1,] != "" ]
    
    # Split data into usefull training predictors and results
    iCol_jnk <- c(1,2,3,4,5,6, 7, dim(data)[2]) # indexes found by inspection of data.
    dataPrd <- data[,-iCol_jnk]     # Predictors
    dataRsp <- data[,dim(data)[2]]  # Responce
```

When developing machine learning systems it is standard practice to split data sets into
training, validation and testing sets.  This is done specifically to prevent overturning 
of the algorithm and to gain confidence in how the system will perform on fresh data.
In our case the testing data has been separately provided so that only training and validation 
data must be formed.  Training data is what the algorithm will directly use to generate
a classification system, once a method has been generated it will be applied to the validation 
data set if performance is not deemed sufficient further refinements can be made in the 
training phase.  Once and only once the system will be applied to the testing data set after the 
developer is satisfied.  Standard practise is to split data ~70% training, 30% validation if 
testing data is separate.


```r
    inTrain <- createDataPartition( y = dataRsp, p=0.7, list=FALSE)
    # Use small set to speed debugging and development
    # inTrain <- createDataPartition( y = dataRsp, p=0.05, list=FALSE)

    dataPrdTraining <- dataPrd[inTrain,]  # training predictors
    dataRspTraining <- dataRsp[inTrain]   # training responces
    dataPrdVal      <- dataPrd[-inTrain,] # validation predictors
    dataRspVal      <- dataRsp[-inTrain]  # validation responces
```

To speed algorithm development all processors on a multi-core computer should be utilized
simultaneously, this is setup below.  Detailed discussion can be found in the linked document
[github](https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md).

While creating the fitControl object we specify that the machine learning method
in caret will use "cv" or Cross Validation with 10-folds.  Cross Validation is a 
process where a number of unique subsets of the training data are used to generate 
different models these are then tested on the remaining data.  


```r
    # Setup parallel cluster to train faster.
    cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
    registerDoParallel(cluster)
    fitControl <- trainControl(method = "cv", number = 10, allowParallel = TRUE)
```

As a first attempt to create an algorithm to classify athletic movement quality we will use
a Random Forest ("RF"), this along with boosting, are commonly the top 
performing methods in prediction contests.


```r
    fit <- train(x = dataPrdTraining, y = dataRspTraining, 
                 method="rf", trControl = fitControl)
```

```
## Random Forest 
## 
## 13453 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 12109, 12109, 12107, 12106, 12109, 12108, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9933101  0.9915363
##   27    0.9918230  0.9896551
##   52    0.9832008  0.9787454
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

To better visualize the quality of our first modeling attempt we will cross plot true
classifications and our models predicted classifications, ideally all non-zero numbers
will be along the diagonal.  As can be seen we were able to fit the training data well.


```r
    ptm <- proc.time() # Used to time predict call duration
    prdTrOut <- predict(fit, dataPrdTraining)
    time_Training <- proc.time() - ptm # record predict call duration
    print( table(prdTrOut, dataRspTraining) )
```

```
##         dataRspTraining
## prdTrOut    A    B    C    D    E
##        A 3830    0    0    0    0
##        B    0 2603    0    0    0
##        C    0    0 2347    0    0
##        D    0    0    0 2203    0
##        E    0    0    0    0 2470
```

Next we will test the model with our validation data set, this data was not used
in deriving the classification method.  As can be seen our model also predicts this 
data set well.


```r
    ptm <- proc.time() # Used to time predict call duration
    prdValOut <- predict(fit, dataPrdVal)
    time_Val <- proc.time() - ptm # record predict call duration
    print( table(prdValOut, dataRspVal) )
```

```
##          dataRspVal
## prdValOut    A    B    C    D    E
##         A 1639    7    0    0    0
##         B    2 1106    9    0    0
##         C    0    2  996   30    6
##         D    0    0    0  914    5
##         E    0    0    0    0 1047
```

Here we quantify how well our model predicted the validation data set, where 1.0 would be
perfectly correct classification.  


```r
    acc_testing <- postResample(dataRspVal, prdValOut)
    print(paste0("Model accuracy (validation): ", round(acc_testing[[1]],5) ))
```

```
## [1] "Model accuracy (validation): 0.98942"
```

Our above results are promising but given that we are using over 20 predictors and have
thousands of data points it is possible that our model, while accurate, is too cumbersome 
to use in a real world system.  To test this we look at how long our algorithm took 
to output each sample prediction we have run so far.  Sensors collecting data were 
reported to be running at 45 Hz, so if our algorithm can not return 45 classifications 
per second utilizing it in a real-time system may not be possible.  As can be seen 
our method executes extremely fast.


```r
    timePredTotal   <- time_Training + time_Val
    secPerSamp      <- timePredTotal[3] / (dim(dataPrdTraining)[1] + dim(dataPrdVal)[1])
    # Data collection rate of 45 samples per second
    sec2Make45Predictions <- 45*secPerSamp  # samples * (sec/sample) 
```

```
## [1] "Calculation duration (sec) per sample: 5.3e-05"                 
## [2] "Seconds need to make one seconds worth of predictions: 0.002365"
```

Finally our method is applied to the testing data set for which we do not have 
truth classifications.  Given the very high accuracy of our model on the validation 
set we are confident it will also work well on the testing set.


```r
    # Code copied from above to import and evaluate testing data set
    fname <- "pml-testing.csv"
    data <- read.csv(file = fname, header = TRUE, sep = ",", stringsAsFactors = TRUE)
    
    # format imported data
    data        <- data[ data$new_window == "no", ]
    data        <- data[ colSums(!is.na(data)) > 0]
    data        <- data[ , data[1,] != "" ]
    # iCol_jnk    <- c(1,2,3,4,5,6, 7, dim(data)[2])
    iCol_jnk    <- c(1,2,3,4,5,6,7) # testing data does not have responce info
    dataPrd_qz  <- data[,-iCol_jnk]        # predictors
    # dataOut_qz  <- data[,dim(data)[2]]   # responce    
    predTestOut <- predict(fit, dataPrd_qz)
    print(predTestOut)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

## Conclusion 
We have developed a system to classify, potentially in real time, the quality of 
an exercise movement.  The above strategy of creating a machine learning classifier
may or may not be expandable to significant numbers of other exercises but as a first
attempt at the problem it was found to work well.  





