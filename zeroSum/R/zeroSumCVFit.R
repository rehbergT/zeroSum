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
#' @useDynLib zeroSum
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
            polish = 0,
            devianceStop = 0.1 )
{
    # some basic checks for the passed arguments
    checkNumericMatrix(x, 'x')
    
    checkType( type )
    if( type == "elNet" || type == "zeroSumElNet" ){
        checkNumericVector(y, 'y')
    } else if( type == "zeroSumLogistic" )
    {
        checkBinominalVector(y, 'y')
    }

    if( nrow(x) != length(y) )
    {
        stop("number of rows of X does not match length of Y!\n")
    }

    checkNumericVector( lambdaSequence, "lambdaSequence")
    checkDouble( alpha, "alpha")
    
    checkAlgo(algorithmCV, "algorithmCV")
    checkAlgo(algorithmAllSamples, "algorithmAllSamples")
   
    checkDouble( epsilon, "epsilon")
    checkDouble( lambdaSteps, "lambdaSteps")
    checkInteger( nFold, "nFold")

    if( nFold > nrow(x)   ) 
    {
        stop("nFold bigger than sample size\n")
    }

    if( devianceStop==FALSE )
    {
        devianceStop = 0
    } else if( devianceStop==FALSE )
    {
        devianceStop = 0.1
    }
    checkDouble( devianceStop, "devianceStop")
    if( devianceStop < 0 || devianceStop >1 )
    {
        stop("devianceStop is not within [0,1]\n")
    }


    devianceStop <- lambdaSteps * devianceStop

    # Sample size and feature size
    N <- nrow(x)
    P <- ncol(x)

    # determine lambdaMax
    if( length(lambdaSequence) == 1 & lambdaSequence[1] == 0 )
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


        if( type == "elNet" )
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
            lambdaMax <- .Call("LambdaMax", x, beta0, alpha)
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
        lambdaSeq <- sort( lambdaSequence, decreasing=TRUE)
        lambdaSteps <- length(lambdaSeq)
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

    if( is.null(foldid) )
    {
        foldid <- sample(rep( rep(1:nFold), length.out=N))
    } else 
    {
        if( length(foldid) != N )
            stop("invalid fold numbering (( length(foldid) != N ))\n")
        nFold <- max(foldid)
    }
    
    testX <- list()    
    for( i in 1:nFold )
    {
        testX[[i]] <- which( foldid==i )
        if( length(testX[[i]])==0 )
            stop("invalid fold numbering\n")
    }

    if(verbose)
    {
        cat("Cross validation subsamples\n")
        print(testX)
        if(any(duplicated( unlist(testX))))
            stop("Duplicated samples in CV\n")

        cat("nFolds:")
        print(nFold)
    }


    logLikeNull   <- NULL
    logLikeSat    <- NULL
    
    if( type == "elNet" || type == "zeroSumElNet" )
    {
        tmp <- rep( 0, N )
        for(i in 1:nFold)
        {
            betaNull <- rep(0,P)
            betaNull[1] <- mean( y[ -testX[[i]] ] )
            tmp[ testX[[i]] ] <- y[ testX[[i]] ] - x[ testX[[i]], ] %*% betaNull
        }
        logLikeNull <- -sum( ( tmp )^2 )   

        betaSat <- rep(0,P)
        zeroSumRegression( x, y, betaSat, lambdaSeq[lambdaSteps]*0.01, 
                alpha, offset, type, algorithmCV, verbose=FALSE, 
                precisionCV, FALSE, 1 )
        logLikeSat <- -vectorElNetCostFunction( x, y, betaSat, 0, 0 )$rss
        
    }

    nullDeviance <- ( logLikeSat - logLikeNull )
    
    if(verbose)
    {
        cat("logLikeSat\n")
        print(logLikeSat)
        cat("logLikeNull\n")
        print(logLikeNull)
    }

    devianceRatio     <- list()
    logLikelihood     <- list()
    logLikelihoodCV   <- list()
    logLikelihoodCVSD <- list()
    numberOfBetas     <- list()
    beta              <- list()   
    for(i in 1: (nFold+1) )
    {
        beta[[i]] <- rep(0.0, P)
    }

    coefs <- matrix(0, ncol=P, nrow=0)

    if(verbose) cat("lambdaSequence.\n")

    for( k in 1 : lambdaSteps)
    {
        if(verbose){
            print(sprintf("Lambda Step: %d",k))
        }

        if(parallel==TRUE) 
        {
            output <- foreach(i = 1:(nFold+1) ) %dopar% 
            {
                loopbeta <- beta[[i]]
                if(i != nFold+1 )
                {
                    zeroSumRegression( 
                        x[-testX[[i]],], y[-testX[[i]]], loopbeta, lambdaSeq[k], alpha,
                        offset, type, algorithmCV, verbose=FALSE, precisionCV, 
                        diagonalMoves, polish )
                } else
                {
                    zeroSumRegression( 
                        x, y, loopbeta, lambdaSeq[k], alpha, offset, type, algorithmAllSamples,
                        verbose=FALSE, precisionCV, diagonalMoves, polish )
                }
                loopbeta
            }
        } else
        {
            output <- foreach(i = 1:(nFold+1) ) %do% 
            {
                loopbeta <- beta[[i]]
                if(i != nFold+1 )
                {
                    zeroSumRegression( 
                        x[-testX[[i]],], y[-testX[[i]]], beta[[i]], lambdaSeq[k], alpha,
                        offset, type, algorithmCV, verbose=FALSE, precisionCV, 
                        diagonalMoves, polish )
                } else
                {
                    zeroSumRegression( 
                        x, y, beta[[i]], lambdaSeq[k], alpha, offset, type, algorithmAllSamples,
                        verbose=FALSE, precisionCV, diagonalMoves, polish )
                }
                loopbeta
            }
        }
        beta <- output
        coefs <- rbind(coefs, beta[[ nFold+1 ]] )
        
        numberOfBetas[[k]] <- sum( beta[[ nFold+1 ]][-1] != 0 )

        if( type == "elNet" || type == "zeroSumElNet" )
        {  
            tmp <- rep( 0, nFold )
            for(i in 1:nFold)
            {
                tmp[i] <- mean( (y[ testX[[i]] ] - x[ testX[[i]], ] %*% beta[[i]])^2 )
            }
            logLikelihood[[k]]     <- -vectorElNetCostFunction( x, y, beta[[ nFold+1 ]], 0, 0 )$rss /N
            logLikelihoodCV[[k]]   <- -mean(tmp)
            logLikelihoodCVSD[[k]] <- sd(tmp)/( sqrt( length(tmp) ))              
        }
              

        devianceRatio[[k]] <- 1 - ( logLikeSat - logLikelihoodCV[[k]] ) / nullDeviance
               
        if( devianceStop != 0 & devianceStop( unlist(devianceRatio), devianceStop) )
        {            
            break            
        }
    }

    
    lambdaSteps <- length(devianceRatio)
    lambdaSeq   <- lambdaSeq[1:lambdaSteps]

    numberOfBetas <- unlist(numberOfBetas)
    
    logLikelihood     <- unlist(logLikelihood)
    logLikelihoodCV   <- unlist(logLikelihoodCV)
    logLikelihoodCVSD <- unlist(logLikelihoodCVSD)
    
    
    devianceRatio <- unlist(devianceRatio)
    devianceRatio[ devianceRatio < 0 | is.infinite(devianceRatio) ] <- 0
    
    maxLogLikeCV <- max( logLikelihoodCV )
    lambdaMin    <- which( logLikelihoodCV  == maxLogLikeCV )[1]
    maxCV_SD     <- maxLogLikeCV - logLikelihoodCVSD[ lambdaMin ]
    
    ## find lambda where the CVerror is as close as possible to minCV + SD
    distances <- abs( logLikelihoodCV[1:lambdaMin] - maxCV_SD)
    distFrom_minCV_SD <- min( distances )
    lambda1SE <- which( distances == distFrom_minCV_SD )
    
    if(verbose)
    {
        print(sprintf( "LambdaMin: %e, Lambda1SE: %e", lambdaSeq[lambdaMin], lambdaSeq[lambda1SE]))
    }
      

    colnames(coefs) <- colnames(x)

    fitresult <- zeroSumCVFitObject(    lambdaSeq,
                                        alpha,
                                        devianceRatio, 
                                        type, 
                                        algorithmCV, 
                                        algorithmAllSamples,
                                        coefs,
                                        lambdaMin,
                                        lambda1SE,
                                        offset, 
                                        precisionCV, 
                                        diagonalMoves, 
                                        polish, 
                                        numberOfBetas,
                                        logLikelihoodCV,
                                        logLikelihood,
                                        logLikelihoodCVSD )
    return(fitresult)


}
