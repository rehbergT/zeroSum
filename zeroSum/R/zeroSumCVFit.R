#' Description of zeroSumCVFit function
#'
#' This function determines for a given dataset x, y and elastic net parameter
#' alpha an optimal lambda value by cross validation (cv). It returns
#' a linear model for the optimal lambda. An appropiate lambda sequence 
#' is estimated or can be passed as an argument. For each lambda value
#' an nFold cv error is calculated. The optimal lambda corresponds to the lowest
#' cv error.
#'
#' @param x data as a numeric matrix object (rows=samples). 
#'          The zero-sum regression requires data on the log scale, i.e.
#'          x should be log-transformed data.
#'
#' @param y response vector to be predicted by x (length(y)==nrow(x))
#'
#' @param lambdaSequence sequence of lambda values to be tested.
#'          If lambda==0 a sequence will be approximated
#'
#' @param alpha Lasso/Ridge adjustment: For alpha = 0 the elastic net becomes
#'              a ridge regularization, for alpha = 1 the elastic net becomes
#'              the lasso regularization
#'
#' @param epsilon If a lambda sequence is estimated, lambdaMax is chosen such
#'              that all coefficients become zero, i.e. lambdaMax is the upper
#'              bound of the lambda sequence. The lower bound is calculated by
#'              lambdaMin = lambdaMax * epsilon and can be adjusted by this
#'              parameter.
#'
#' @param lambdaSteps this parameters determines the number of lambda steps between
#'                    lambdaMin and lambdaMax, i.e higher values for lambdaSteps
#'                    increase the resolution of the regularization path.
#'
#' @param nFold the number of folds used for the cross validation
#'
#' @param foldid allows to determine the folds used for cross validation.
#'
#' @param offset determines if an offset should be used in the
#'               model or not (TRUE/FALSE)
#'
#' @param parallel The cross validation is done within a foreach loop which can be
#'                 executed in parallel. 'doMC' or equivalent needs to be
#'                 registered before using this!
#'
#' @param verbose verbose = TRUE enables output
#'
#' @param type choose the regression type: elNet, zeroSumElNet
#'
#' @param algorithmCV determines the algorithm used for the cross validation:
#'            CD = Coordinate descent (very fast, not so accurate),
#'            CD+LS = Coordinate descent + local search (fast, very accurate),
#'            LS = local search (slow, accurate),
#'            SA = simulated annealing (very slow, very accurate)
#'
#' @param algorithmAllSamples determines the algorithm used for the creation of 
#'           of the final models (lambda.min and lambda.1SE):
#'            CD = Coordinate descent (very fast, not so accurate),
#'            CD+LS = Coordinate descent + local search (fast, very accurate),
#'            LS = Local search (slow, accurate),
#'            SA = Simulated annealing (very slow, very accurate)
#'
#' @param precisionCV stopping criterion of the used algorithms for the CV fit.
#'                    Determines how small the improvement of the cost function
#'                    has to be to stop the algorithm. Default is 1e-6.
#'
#' @param precisionAllSamples stopping criterion of the used algorithms for 
#'            the creation of the final models (lambda.min and lambda.1SE).
#'            Default is 1e-6.
#'
#' @param diagonalMoves allows the CD to use diagonal moves
#'
#' @param lambdaScaler allows to adjust the approximated lambdaMax
#'
#' @param polish enables a local search at the end of CD to polish the result
#'
#' @param devianceStop stops the CV progress if the model's deviance becomes worse
#'                  for lower lambda values. The number of worse deviance
#'                  values which is tolerated is calculated by multiplying
#'                  the lambdaSteps (Default: 100) with the devianceStop
#'                  (Default: 0.1). Use devianceStop = 0 or FALSE to deactivate
#'                  the devianceStop.
#'
#' @return zeroSumCVFitObject
#'
#' @examples
#' set.seed(1)
#' data <- simulateData()
#' fit <- zeroSumCVFit( data$x, data$y, alpha=1)
#' plot( fit, "test")
#' coef(fit, s="lambda.min")
#'
#' @import foreach stats grDevices
#'
#' @export
zeroSumCVFit <- function(
            x,
            y,
            lambdaSequence = 0,
            alpha = 1,
            epsilon = 0.001,
            lambdaSteps = 100,
            nFold = 10,
            foldid = NULL,
            offset = TRUE,
            parallel = FALSE,
            verbose = FALSE,
            type = "zeroSumElNet",
            algorithmCV = "CD",
            algorithmAllSamples = "CD+LS",
            precisionCV = 1e-6,
            precisionAllSamples = 1e-6,
            diagonalMoves = TRUE,
            lambdaScaler = 1,
            polish = 100,
            devianceStop = 0.1 )
{
    # some basic checks for the passed arguments
    if( class(x) != "matrix" | typeof(x) != "double"  )
    {
        stop("type of passed x is not a numeric matrix\n")
    }

    if( (class(y) != "numeric" | typeof(y) != 'double') &&
        (class(y) != "integer" | typeof(y) != 'integer')  )
    {
        stop("type of passed y is not numeric or integer\n")
    }

    if( nrow(x) != length(y) )
    {
        stop("number of rows of X does not match length of Y!\n")
    }

    if( class(lambdaSequence) != "numeric" |
        typeof(lambdaSequence) != 'double'   )
    {
        stop("type of passed lambda sequence is not numeric\n")
    }

    if( length(lambdaSequence) < 2 & lambdaSequence != 0  )
    {
        stop("only one lambda in lambdasequence is passed\n")
    }

    if( class(alpha) != "numeric" | typeof(alpha) != 'double'   )
    {
        stop("type of passed alpha is not numeric\n")
    }
    if( class(algorithmCV) != "character" & typeof(algorithmCV) != "character" |
        (   algorithmCV != "CD"   &
            algorithmCV != "SA"   &
            algorithmCV != "LS"   &
            algorithmCV != "CD+LS"))
    {
            cat( "Selected algorithmCV is not valid\n")
            cat( "Now using CD\n" )
            algorithmCV <- "CD"
    }
    if( class(algorithmAllSamples) != "character" & 
        typeof(algorithmAllSamples) != "character" |
        (   algorithmAllSamples != "CD"   &
            algorithmAllSamples != "SA"   &
            algorithmAllSamples != "LS"   &
            algorithmAllSamples != "CD+LS"))
    {
            cat( "Selected algorithmAllSamples is not valid\n")
            cat( "Now using CD+LS\n" )
            algorithmAllSamples <- "CD+LS"
    }

    if( class(type) != "character" & typeof(type) != "character" |
        (   type != "elNet" &
            type != "zeroSumElNet"))
    {
            cat( "Selected type is not valid\n")
            cat("Use zeroSumElNet (default), zeroSumLogistic or elNet\n")
            type ="zeroSumElNet"
    }

    if( class(epsilon) != "numeric" | typeof(epsilon) != 'double'   ) 
    {
        stop("type of epsilon for calculating lambda min is not numeric\n")
    }

    if( class(lambdaSteps) != "numeric" | typeof(lambdaSteps) != 'double'   ) 
    {
        stop("type of lambdaSteps for calculating lambda min is not numeric\n")
    }

    if( class(nFold) != "numeric" | typeof(nFold) != 'double'   ) {
        stop("type of passed nFold for calculating lambda min is not numeric\n")
    }

    if( nFold > nrow(x)   ) 
    {
        stop("nFold bigger than sample size\n")
    }

    if( devianceStop==FALSE )
    {
        devianceStop = 0
    }

    if( typeof(devianceStop) != "double" || class(devianceStop) != "numeric")
    {
        stop("devianceStop not numeric\n")
    }

    if( devianceStop < 0 || devianceStop >1 )
    {
        stop("devianceStop is not within [0,1]\n")
    }

    devianceStop <- lambdaSteps*devianceStop

    # Sample size and feature size
    N <- nrow(x)
    P <- ncol(x)

    # determine lambdaMax
    if( lambdaSequence == 0 )
    {
        # in the ridge case (alpha==0) lambdaMax can not be determined
        # therefore a small alpha is used to determine a lambda max
        # the variable ridge is used as a bool to revert alpha to zero
        # after the calculation

        if(verbose)
        {
            cat("Determine lambdaMax\n")            
        }

        ridge <- FALSE
        if(alpha==0)
        {
            alpha <- 0.1
            ridge <- TRUE
        }
        lambdaMax <- NULL


        if( type == "elNet")
        {
            beta0 <- mean(y)
            res <- rep( 0, P )
            for(j in 1 : P)
            {
                for( i in 1 : N )
                {
                    res[j] <- res[j] +  x[i,j] * (y[i]-beta0)
                }
            }
            lambdaMax <- max(abs(res)) / ( N * alpha )

        } else if( type == "zeroSumElNet")
        {
            beta0 <- y - mean(y)
            maxRes <- -.Machine$double.xmax
            ind <- 1

            for(k in 1:P)
            {
                for(s in 1:k)
                {
                    if(s==k) next
                    a <- abs(sum( ( x[,s] - x[,k] ) * beta0 ))                   
                    if( a > maxRes ) maxRes <- a
                }
            }
            lambdaMax <- maxRes / ( 2.0 * N * alpha )

        }
        lambdaMax <- lambdaMax * lambdaScaler
        # lambdaMin is calculated with the epsilon paramter
        lambdaMin <- epsilon * lambdaMax

        # the lambda sequence is constructed by equally distributing lambdaSteps
        # value on the lineare log scale between lambdaMin and lambdaMax
        lambdaSeq <- exp(   seq(log(lambdaMax), 
                            log(lambdaMin),
                            length.out = lambdaSteps))

        # revert alpha to zero in the ridge case
        if(ridge==TRUE)
        {
            alpha <- 0
        }


    }
    else
    {
        lambdaSeq <- lambdaSequence
    }

    if(verbose)
    {
        cat("Lambda sequence")
        print(lambdaSeq)
    }

    # if x has no colnames numerate them
    if(is.null(colnames(x)))
    {
        tmp <-c("Intercept", seq(1, ncol(x)))
    }else{
        tmp <-c("Intercept", colnames(x) )
    }

    x <- cbind( rep( 1.0, nrow(x)), x)
    colnames(x) <- tmp

    N <- nrow(x)
    P <- ncol(x)

    testX <- list()
    if(!is.null(foldid)){
        nFold <- max(foldid)
        for( i in 1:nFold )
        {
            testX[[i]] <- which( foldid==i )
            if( length(testX[[i]])==0 )
                stop("invalid fold numbering\n")
        }

    }else{
        # create subsamples (test samples)
        # training sample are determined by leaving out the test samples:
        #      data[  -test samples] )
        remainder <- N %% nFold
        testsize <- floor(N / nFold)
        randomIndices <- sample( 1 : N, N)

        j <- 1
        for( i in 1:nFold )
        {
                if( i <= remainder)
                    testX[[i]] <- randomIndices[ j : (j+testsize) ]
                else
                    testX[[i]] <- randomIndices[ j : (j+testsize-1)]

                j <- j + testsize
        }
    }


    if(verbose)
    {
        cat("Crossvalidiation subsamples\n")
        print(testX)
    }

    devianceRatio <- list()

    tmp <- rep(0,N)
    for(i in 1:nFold)
    {
        betaNull <- rep(0,P)
        betaNull[1] <- mean(y[-testX[[i]]])

        tmp[ testX[[i]] ] <-  x[ testX[[i]], ] %*% betaNull
    }
    devNull <- mean(( y - tmp )^2)
    betaSat <- betaNull
    zeroSumRegression( x, y, betaSat, lambdaSeq[lambdaSteps]*0.01, 
            alpha, offset, type, algorithmCV, verbose=FALSE, 
            precisionCV, FALSE, 1 )

    devSat <- mean(( y - x %*% betaSat )^2)

    meanCVE <- list()
    CV_SD <- list()
    numberOfBetas <- list()

    beta <- list()
    for(i in 1:nFold)
    {
        beta[[i]] <- rep(0.0, P)
    }

    if(verbose)
    {
        cat("devSat\n")
        print(devSat)
        cat("devNull\n")
        print(devNull)
    }

    if(verbose) cat("lambdaSequence.\n")

    for( k in 1 : lambdaSteps)
    {
        if(verbose){
            print(sprintf("Lambda Step: %d",k))
        }

        tmp <- rep( 0, nFold)

        if(parallel==TRUE) 
        {
            output <- foreach(i = 1:nFold ) %dopar% 
            {
                zeroSumCVFit_subsetFit( beta[[i]], x, y, testX[[i]],
                            lambdaSeq[k], alpha, offset, type, algorithmCV, 
                            precisionCV, diagonalMoves, polish)  
            }
        } else
        {
            output <- foreach(i = 1:nFold ) %do% 
            {
                zeroSumCVFit_subsetFit( beta[[i]], x, y, testX[[i]],
                            lambdaSeq[k], alpha, offset, type, algorithmCV, 
                            precisionCV, diagonalMoves, polish)  
            }
        }
        
        numberOfCVBetas <- rep(0,nFold)

        for(i in 1:nFold)
        {
            numberOfCVBetas[i] <- sum( beta[[i]][-1] != 0 )
        }

        numberOfBetas[[k]] <- round(mean(numberOfCVBetas))
        if( type == "zeroSumElNet" && numberOfBetas[[k]] == 1 )
        {
            if( mean(numberOfCVBetas) > 1 )
            {
                numberOfBetas[[k]] <- 2
            }else
            {
                numberOfBetas[[k]] <- 0
            }
        }

        tmp <- rep( 0, nFold)
        for(i in 1:nFold)
        {
            tmp[i] <- output[[i]]$mean
        }
        meanCVE[[k]] <- mean(tmp)
        CV_SD[[k]] <-  sd(tmp)/( sqrt( length(tmp) ))

        ratio <- ( devSat - meanCVE[[k]] ) / ( meanCVE[[k]] - devNull )
        devianceRatio[[k]] <- 1 - ratio

        if( devianceStop!=0 & k >= devianceStop )
        {
            test <- 0
            i <- k
            while(i>1)
            {
                if( devianceRatio[[i]] <  devianceRatio[[i-1]] )
                {
                    test <- test+1
                }
                i <- i-1
            }

            if( test > devianceStop)
            {
                break
            }
        }
    }

    lambdaSteps <- length(meanCVE)
    lambdaSeq <- lambdaSeq[1:lambdaSteps]

    numberOfBetas <- unlist(numberOfBetas)
    cv_error <- unlist(meanCVE)
    cv_error_sd <- unlist(CV_SD)
    devianceRatio <- unlist(devianceRatio)
    devianceRatio[ devianceRatio < 0 | is.infinite(devianceRatio) ] <- 0

    minCV <- min(cv_error)
    lambdaMin <- which( cv_error == minCV )[1]
    minCV_SD <- minCV + cv_error_sd[lambdaMin]

    ## find lambda where the CVerror is as close as possible to minCV + SD
    distFrom_minCV_SD <- min( abs( cv_error[1:lambdaMin] - minCV_SD) )
    lambda1SE <- which( abs( cv_error - minCV_SD ) == distFrom_minCV_SD )

    if(verbose)
    {
        print(sprintf( "LambdaMin: %e, Lambda1SE: %e", lambdaSeq[lambdaMin], lambdaSeq[lambda1SE]))
        cat("calc coefs of lambda1SE\n")     
    }
    
    beta1SE <- rep( 0, P )
    betaMin <- rep( 0, P )
    
    zeroSumRegression( x, y, beta1SE, lambdaSeq[lambda1SE], alpha,
                    offset, type, algorithmAllSamples, 
                    verbose=FALSE, precisionAllSamples,
                    diagonalMoves, polish )
    
    betaMin <- rep( 0, P )
    for(i in 1:P)
        betaMin[i] <- beta1SE[i]
    
    if(verbose) cat("calc coefs of lambdaMin\n")
    
    zeroSumRegression( x, y, betaMin, lambdaSeq[lambdaMin], alpha,
                        offset, type, algorithmAllSamples,
                        verbose=FALSE, precisionAllSamples,
                        diagonalMoves, polish ) 

    names(betaMin) <- colnames(x)
    names(beta1SE) <- colnames(x)

    fitresult <- zeroSumCVFitObject(    lambdaSeq, 
                                        cv_error, 
                                        cv_error_sd, 
                                        alpha,
                                        devianceRatio, 
                                        type, 
                                        algorithmCV, 
                                        algorithmAllSamples,
                                        lambdaMin, 
                                        betaMin, 
                                        lambda1SE, 
                                        beta1SE, 
                                        offset, 
                                        precisionCV, 
                                        diagonalMoves, 
                                        polish, 
                                        numberOfBetas )
    return(fitresult)

}
