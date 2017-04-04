#' Description of regresion object
#'
#' @keywords internal
#'
#' @export
regressionObject <- function(x, y, beta , lambda, alpha, gamma=0.0, cSum=0.0,
        type=zeroSumTypes[1,1], weights=NULL, zeroSumWeights=NULL,
        penalty.factor=NULL, fusion=NULL, precision=1e-8,
        useOffset=TRUE, useApprox=TRUE, downScaler=1, algorithm="CD",
        diagonalMoves=TRUE, polish=10, standardize=TRUE, lambdaSteps=1,
        gammaSteps=1, nFold=1, foldid=NULL, epsilon=0.001, cvStop = 0.1,
        verbose=FALSE, lambdaScaler=1, cores=1 )
{
    dataObject <- list()

    checkType( type )
    id <- which( zeroSumTypes[,1] == type)
    dataObject$type <- as.integer( zeroSumTypes[id,2] )

    dataObject$x <- checkNumericMatrix(x, 'x')
    if(is.null(colnames(dataObject$x)))
        colnames(dataObject$x) <- as.character(seq(1, ncol(dataObject$x)))

    dataObject$y <- checkResponse( y, 'y', type )

    if( nrow(dataObject$x)!= nrow(dataObject$y) )
        stop("nrow(x) != nrow(y) !")

    N <- nrow(dataObject$x)
    P <- ncol(dataObject$x)

    dataObject$K <- NCOL(dataObject$y)
    dataObject$P <- P
    dataObject$N <- N

    dataObject$cSum  <- checkDouble( cSum,  "cSum")
    dataObject$downScaler <- checkDouble( downScaler,  "downScaler")
    dataObject$alpha <- checkDouble( alpha, "alpha")
    dataObject$nFold <- checkInteger( nFold, "nFold")

    if(cores == "auto") {
        dataObject$cores = as.integer(-1)
    } else {
        dataObject$cores = checkInteger( cores, "cores")
    }

    if( is.null(weights)) {
        weights <- rep( 1/N, N)

    } else {
        checkNonNegativeWeights(weights, N, "weights")
    }
    dataObject$w <- weights

    if( is.null(penalty.factor)) {
        penalty.factor <- rep( 1, P)
    } else {
        checkNonNegativeWeights(penalty.factor, P, "penalty.factors")
    }
    dataObject$v <- penalty.factor

    if( is.null(zeroSumWeights)) {
        zeroSumWeights <- rep( 1, P)
    } else {
        checkWeights(zeroSumWeights, P, "zeroSumWeights")
    }

    dataObject$u <- zeroSumWeights

    dataObject$lambdaSteps <- checkInteger( lambdaSteps, "lambdaSteps" )
    dataObject$lambda      <- checkDouble( lambda, "lambda")

    dataObject$gammaSteps  <- checkInteger( gammaSteps, "gammaSteps" )
    dataObject$gamma       <- checkDouble( gamma, "gamma")

    if( dataObject$nFold > N )
        stop("nFold bigger than sample size\n")

    if( is.null(foldid) )
    {
        dataObject$foldid <- sample(rep( rep(1:nFold), length.out=N))
    } else
    {
        if( length(foldid) != N )
            stop("invalid fold numbering (( length(foldid) != N ))\n")
        dataObject$nFold <- max(foldid)
        dataObject$foldid <- foldid
    }


    if( is.null(fusion))
    {
        dataObject$fusion <- NULL
        dataObject$nc     <- as.integer(0)
    } else
    {
        checkSparseMatrix(fusion, "fusion")
        dataObject$fusion <- fusion
        dataObject$nc     <- as.integer(nrow(fusion))
    }

    if( is.null(dataObject$fusion) && type %in% zeroSumTypes[c(5,6,11,12,17,18),1]  )
        stop("no fusion kernel supplied\n")


    if(standardize==TRUE)
    {
        dataObject$standardize = TRUE

        ## we need the sd which is devided by N
        ## to get the same results as the glmnet
        ## -> we have to write our own scale methode
        scaleNew <- function(r,s,c) { (r-c)/s }
        sdNew <- function(v) { sqrt(mean((mean(v)-v)^2)) }

        dataObject$xSD <- apply( x,2,sdNew)
        dataObject$xM  <- colMeans(x)

        dataObject$ySD <- sdNew(y)
        dataObject$yM  <- mean(y)

        if( any(dataObject$xSD==0) ) stop("sd calculation of x created 0")
        if( any(dataObject$ySD==0) ) stop("sd calculation of y created 0")

        dataObject$x <- t(apply( x, 1, scaleNew, dataObject$xSD, dataObject$xM ))

        ## y can only be standardized if it is numeric
        if( type %in% zeroSumTypes[c(1:6),1])
        {
            dataObject$y <- scaleNew(dataObject$y, dataObject$ySD, dataObject$yM)
            dataObject$lambda <- dataObject$lambda / dataObject$ySD
            dataObject$u <- dataObject$ySD / dataObject$xSD

        } else
        {
            dataObject$lambda <- dataObject$lambda
            dataObject$u <- rep(1.0,P) / dataObject$xSD
        }

        if( !is.null(dataObject$fusion) )
        {
            scaler <- dataObject$ySD / dataObject$xSD

            for( i in 1:nrow(dataObject$fusion))
            {
                dataObject$fusion[i,] <- dataObject$fusion[i,] * scaler
            }
        }

    } else
    {
        dataObject$standardize = FALSE
    }

    dataObject$verbose <- as.integer(verbose)

    if( length(dataObject$lambda) == 1 && dataObject$lambdaSteps > 1 )
    {
        # in the ridge case (alpha==0) lambdaMax can not be determined
        # therefore a small alpha is used to determine a lambda max
        # the variable ridge is used as a bool to revert alpha to zero
        # after the calculation
        if(verbose) cat("Determine lambdaMax\n")

        ridge <- FALSE

        if( dataObject$alpha==0.0 )
        {
            dataObject$alpha <- 0.01
            ridge <- TRUE
        }
        lambdaMax <- NULL
        res <- NULL

        if( type %in% zeroSumTypes[1:6,1] )
        {
            nM  <- mean(dataObject$y)
            res <- as.matrix( dataObject$y - nM, ncol=1 ) * dataObject$w
            lambdaMax <- max(abs( t(dataObject$x) %*% res )) / ( min(dataObject$v) * dataObject$alpha )

        }else if( type %in% zeroSumTypes[7:12,1] )
        {
            nM  <- getLogisticNullModel( dataObject$y, dataObject$w, 10)
            res <- as.matrix( ( nM$z - nM$beta0 ) * nM$w, ncol=1 )
            lambdaMax <- max(abs( t(dataObject$x) %*% res )) / ( min(dataObject$v) * dataObject$alpha )

        } else if( type %in% zeroSumTypes[13:18,1] )
        {
            nM <- getMultinomialNullModel( dataObject$y, dataObject$w, 10)
            res <- matrix(0,ncol=ncol(nM$z),nrow=nrow(nM$z) )
            for(i in 1:nrow(res) )
                res[i,] <- ( nM$z[i,] - nM$beta0 ) * nM$w[i,]

            lambdaMax <- max(abs( t(dataObject$x) %*% res )) / ( min(dataObject$v) * dataObject$alpha )
        }

        epsilon      <- checkDouble(epsilon)
        lambdaScaler <- checkDouble(lambdaScaler)


        lambdaMax <- lambdaMax * lambdaScaler

        # lambdaMin is calculated with the epsilon paramter
        lambdaMin <- epsilon * lambdaMax

        # the lambda sequence is constructed by equally distributing lambdaSteps
        # value on the lineare log scale between lambdaMin and lambdaMax
        dataObject$lambda <- exp( seq(log(lambdaMax), log(lambdaMin),
                            length.out = dataObject$lambdaSteps))

        # revert alpha to zero in the ridge case
        if( ridge==TRUE )
        {
            dataObject$alpha <- 0.0
        }


    } else if( length(dataObject$lambda) > 1 )
    {
        dataObject$lambdaSteps <- length(dataObject$lambda)
    }



    if( length(dataObject$gamma) == 1 && dataObject$gammaSteps > 1 )
    {
        print("approx gamma todo")

    } else if( length(dataObject$gamma) > 1 )
    {
        dataObject$gammaSteps <- length(dataObject$gamma)
    }


    if( cvStop==FALSE )
        cvStop = 1.0;

    dataObject$cvStop = checkDouble(cvStop,"CV-Stop")
    if( dataObject$cvStop < 0 || dataObject$cvStop > 1 )
        stop("cvStop is not within [0,1]\n")
    dataObject$cvStop <- as.integer(round( dataObject$lambdaSteps * dataObject$cvStop))

    if( is.null(dataObject$fusion) )
    {
        dataObject$fusionC <- NULL
    } else
    {
        dataObject$fusionC <- as.matrix( Matrix::summary(dataObject$fusion))
        dataObject$fusionC <- dataObject$fusionC[ order(dataObject$fusionC[,2]),]
        dataObject$fusionC[,1:2] <- dataObject$fusionC[,1:2] - 1 ## array index in C starts with 0
    }

    if( is.null(beta) )
    {
        if( type %in% zeroSumTypes[ c(1,3,5,7,9,11),1] )
        {
            beta <- rep( 0, ncol(x)+1 )  ## +1 for the offset

        } else if( type %in% zeroSumTypes[ c(2,4,6,8,10,12),1] )
        {
            beta <- rep( 0, ncol(x)+1 )
            beta[2] <- dataObject$cSum / dataObject$u[1]

        } else if( type %in% zeroSumTypes[c(13,15,17),1] )
        {
            beta <- matrix( 0, ncol=ncol(dataObject$y), nrow=ncol(x)+1 )

        } else if( type %in% zeroSumTypes[ c(14,16,18),1] )
        {
            beta <- matrix( 0, ncol=ncol(dataObject$y), nrow=ncol(x)+1 )
            beta[2,] <- rep( dataObject$cSum / dataObject$u[1], ncol(dataObject$y) )
        }

    } else
    {
        if( type %in% zeroSumTypes[ 1:12,1] )
        {
            if( length(beta) != ncol(x)+1)
            {
                stop("Length of betas doesn't match ncol(x)!")
            }

        } else
        {
            if( nrow(beta) != ncol(x)+1)
            {
                stop("Length of betas doesn't match ncol(x)!")
            }
        }


        if( type %in% zeroSumTypes[c(2,4,6,8,10,12),1] &&
                as.numeric(beta[-1] %*% dataObject$u) != dataObject$cSum )
        {
            stop("Sum of betas doesn't match cSum!")
        }

        if( type %in% zeroSumTypes[c(14,16,18),1] &&
                !(any( as.numeric(beta[-1,] %*% dataObject$u) != dataObject$cSum )) )
        {
            stop("Sum of betas doesn't match cSum!")
        }
    }
    dataObject$beta <- beta

    if(is.null(colnames(x)))
        colnames(x) <- as.character(seq(1,ncol(x)))

    if( type %in% zeroSumTypes[ 1:12,1] )
    {
        names(dataObject$beta) <- c( "Intercept",colnames(x))

    } else
    {
        rownames(dataObject$beta) <- c( "Intercept",colnames(x))
    }

    dataObject$precision   <- checkDouble( precision, "precision")
    dataObject$useOffset   <- checkInteger(useOffset)


    checkAlgo(algorithm)
    id <- which( zeroSumAlgos[,1] == algorithm)
    dataObject$algorithm <- as.integer( zeroSumAlgos[id,2] )


    if( type %in% zeroSumTypes[ c(4,5,6,10,11,12,16,17,18),1] )
    {
         dataObject$algorithm   <- as.integer(3)
    }

    dataObject$useApprox   <- checkInteger(useApprox)

    if( dataObject$type <= 6  )
    {
        dataObject$useApprox <- as.integer(FALSE)

    }else if( dataObject$algorithm==1 )
    {
        dataObject$useApprox <- as.integer(TRUE)
    }

#     if( type %in% zeroSumTypes[ c(4,10,16),1] )
#     {
#          stop("Not implemented yet!")
#     }

    dataObject$diagonalMoves <- checkInteger(diagonalMoves)
    dataObject$polish        <- checkInteger(polish)

    return(dataObject)
}
