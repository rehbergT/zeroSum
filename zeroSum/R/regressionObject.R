#' Description of regresion object
#'
#' @importFrom stats rnorm sd weighted.mean
#'
#' @useDynLib zeroSum
#'
#' @keywords internal
#'
#' @export
regressionObject <- function(x, y, beta , lambda, alpha, gamma=0.0,
        type=zeroSumTypes[1,1], weights=NULL, penalty.factor=NULL,
        fusion=NULL, precision=1e-8, useOffset=TRUE, useApprox=TRUE,
        downScaler=1, algorithm="CD", diagonalMoves=TRUE, polish=TRUE,
        standardize=TRUE, lambdaSteps=1, gammaSteps=1, nFold=1, foldid=NULL,
        epsilon=NULL, cvStop = 0.1, verbose=FALSE, cores=1 )
{
    dataObject <- list()

    checkType( type )
    id <- which( zeroSumTypes[,1] == type)
    dataObject$type <- as.integer( zeroSumTypes[id,2] )

    tmp <- checkData(x, y, weights, type )
    dataObject$x <- tmp$x
    dataObject$y <- tmp$y
    dataObject$w <- tmp$w

    if( !is.null(tmp$ord) )
    {
        dataObject$x <- dataObject$x[ tmp$ord, ]
        dataObject$y <- dataObject$y[ tmp$ord, , drop=FALSE ]
        dataObject$w <- dataObject$w[ tmp$ord ]
    }

    N <- nrow(dataObject$x)
    P <- ncol(dataObject$x)

    dataObject$K <- NCOL(dataObject$y)
    dataObject$P <- P
    dataObject$N <- N

    dataObject$cSum  <- 0.0
    dataObject$downScaler <- checkDouble( downScaler,  "downScaler")
    dataObject$alpha <- checkDouble( alpha, "alpha")
    dataObject$nFold <- checkInteger( nFold, "nFold")

    if(cores == "auto") {
        dataObject$cores = as.integer(-1)
    } else {
        dataObject$cores = checkInteger( cores, "cores")
    }

    if( type %in% zeroSumTypes[13:16,1] ) {
        dataObject$K <- 1
        useOffset <- FALSE
        dataObject$status <- tmp$status
        if( !is.null(tmp$ord) )
            dataObject$status <- dataObject$status[ tmp$ord ]
    }

    if( is.null(penalty.factor)) {
        penalty.factor <- rep( 1, P)
    } else {
        checkNonNegativeWeights(penalty.factor, P, "penalty.factors")
    }
    dataObject$v <- penalty.factor
    dataObject$u <- rep( 1, P)

    dataObject$lambdaSteps <- checkInteger( lambdaSteps, "lambdaSteps" )
    dataObject$lambda      <- checkDouble( lambda, "lambda")

    dataObject$gammaSteps  <- checkInteger( gammaSteps, "gammaSteps" )
    dataObject$gamma       <- checkDouble( gamma, "gamma")

    if( dataObject$nFold > N ){
        print(paste0("more CV folds (default nFold=10) than sample size! Setting nFold to ", N))
        dataObject$nFold <- N
    }

    if( is.null(foldid) )
    {
        if( type %in% zeroSumTypes[13:16,1] ) {
            nonCensored <- sum( dataObject$status==1 )

            if( nonCensored < nFold ){
                print(paste0("more CV folds (default nFold=10) than non-censored ! Setting nFold to ", nonCensored))
                dataObject$nFold <- nonCensored
            }

            dataObject$foldid <- rep(0,N)
            dataObject$foldid[ dataObject$status==1 ] <- sample(rep( rep(1:dataObject$nFold),
                                            length.out=nonCensored))
            dataObject$foldid[ dataObject$status==0 ] <- sample(rep( rep(1:dataObject$nFold),
                                            length.out=dataObject$N-nonCensored))
        } else {
            dataObject$foldid <- sample(rep( rep(1:dataObject$nFold), length.out=N))
        }
    } else
    {
        if( !is.null(tmp$ord) )
            foldid <- foldid[ tmp$ord ]

        if( length(foldid) != N )
            stop("invalid fold numbering (( length(foldid) != N ))\n")

        dataObject$nFold <- max(foldid)
        dataObject$foldid <- foldid
    }

    dataObject$foldid <- as.integer(dataObject$foldid)
    dataObject$nFold  <- as.integer(dataObject$nFold)

    if( dataObject$nFold != 0 )
        checkFolds( dataObject$status, dataObject$foldid, dataObject$nFold, dataObject$type )

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

    if( is.null(dataObject$fusion) && type %in% zeroSumTypes[c(3,4,7,8,11,12,15,16),1] )
        stop("no fusion kernel supplied\n")


    ## calculate the weighted means
    dataObject$xM  <- as.numeric( (t(dataObject$x) %*% dataObject$w) / sum(dataObject$w) )
    dataObject$yM  <- as.numeric( (t(dataObject$y) %*% dataObject$w) / sum(dataObject$w) )

    if(standardize==TRUE)
    {
        dataObject$standardize = TRUE

        sdFunction <- function(v,w) {
            wm <- weighted.mean(v,w)
            return(sqrt( weighted.mean( (v-wm)^2, w) ))
        }

        dataObject$xSD <- apply( dataObject$x, 2, sdFunction, dataObject$w)
        dataObject$ySD <- apply( dataObject$y, 2, sdFunction, dataObject$w)

        if( any(dataObject$xSD==0) ) stop("sd calculation of x created 0")
        if( any(dataObject$ySD==0) ) stop("sd calculation of y created 0")

    } else
    {
        dataObject$standardize = FALSE

        dataObject$xSD <- rep( 1, dataObject$P )
        dataObject$ySD <- rep( 1, dataObject$K )
    }

    if( !(type %in% zeroSumTypes[1:4,1]) )
    {
        dataObject$ySD <- rep( 1, dataObject$K )
        dataObject$yM  <- rep( 0, dataObject$K )
    }

    dataObject$useOffset <- checkInteger(useOffset)
    dataObject$calcOffsetByCentering <- FALSE

    if( dataObject$standardize==TRUE )
    {
        scaleFunction <- function(r,s,c) { (r-c)/s }
        dataObject$x <- t(apply( dataObject$x, 1, scaleFunction, dataObject$xSD, dataObject$xM ))

        if( type %in% zeroSumTypes[1:4,1] )
        {
            dataObject$y <- scaleFunction(dataObject$y, dataObject$ySD, dataObject$yM)
            dataObject$lambda <- dataObject$lambda / dataObject$ySD
            dataObject$u      <- dataObject$u * ( dataObject$ySD / dataObject$xSD )
        } else
        {
            dataObject$u  <- dataObject$u / dataObject$xSD
        }

        if( !is.null(dataObject$fusion) )
        {
            scaler <- dataObject$ySD / dataObject$xSD

            for( i in 1:nrow(dataObject$fusion))
            {
                dataObject$fusion[i,] <- dataObject$fusion[i,] * scaler
            }
        }
    } else if ( dataObject$useOffset==1 )
    {
        centerFunction <- function(r,c) { (r-c) }
        dataObject$x <- t(apply( dataObject$x, 1, centerFunction, dataObject$xM ))

        if( type %in% zeroSumTypes[1:4,1] )
        {
            dataObject$y <- dataObject$y - dataObject$yM

            # calculating an offset is unnecessary if centering is done
            dataObject$useOffset <- as.integer(FALSE)
            dataObject$calcOffsetByCentering <- TRUE
        }
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
        nM  <- NULL
        res <- NULL

        if( type %in% zeroSumTypes[1:4,1] )
        {
            nM  <- sum(dataObject$y * dataObject$w )
            res <- as.matrix( dataObject$y - nM, ncol=1 ) * dataObject$w

        }else if( type %in% zeroSumTypes[5:8,1] )
        {
            nM  <- getLogisticNullModel( dataObject$y, dataObject$w )
            res <- as.matrix( ( nM$z - nM$beta0 ) * nM$w, ncol=1 )

        } else if( type %in% zeroSumTypes[9:12,1] )
        {
            nM <- getMultinomialNullModel( dataObject$y, dataObject$w, 10)
            res <- matrix(0,ncol=ncol(nM$z),nrow=nrow(nM$z) )
            for(i in 1:nrow(res) )
                res[i,] <- ( nM$z[i,] - nM$beta0 ) * nM$w[i,]

        }  else if( type %in% zeroSumTypes[13:16,1] )
        {
            res <- getCoxNullModel( dataObject$y, dataObject$status, dataObject$w )
        }

        if( type %in% zeroSumTypes[seq(1,16,2),1] )
        {
            lambdaMax <- max((abs( t(dataObject$x) %*% res ) / (dataObject$v * dataObject$alpha ))[dataObject$v != 0])
        } else
        {
            lambdaMax <- .Call( "lambdaMax", dataObject$x, res, dataObject$u, dataObject$v, dataObject$alpha, PACKAGE="zeroSum")
        }

        if( !is.null(epsilon) )
        {
            epsilon <- checkDouble(epsilon)
        } else {
            if( N < P ) {
                epsilon <- 0.01
            } else {
                epsilon <- 0.0001
            }

        }

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
        cols <- dataObject$K
        if(type %in% zeroSumTypes[13:16,1]) cols <- 1
        beta <- matrix( 0, ncol=cols, nrow=dataObject$P+1 )

        if( type %in% zeroSumTypes[seq(2,16,2),1] )
        {
            beta[2,] <- rep( dataObject$cSum / dataObject$u[1], cols )
        }

    } else
    {
        beta <- as.matrix(beta)
        if( nrow(beta) != ncol(x)+1)
        {
            stop("Length of betas doesn't match ncol(x)!")
        }
        if( type %in% zeroSumTypes[seq(2,16,2),1] )
        {
            if( !(any( as.numeric(beta[-1,] %*% dataObject$u) != dataObject$cSum )) )
            {
                stop("Sum of betas doesn't match cSum!")
            }
        }

    }

    dataObject$beta <- beta
    rownames(dataObject$beta) <- c( "Intercept",colnames(dataObject$x))

    dataObject$precision   <- checkDouble( precision, "precision")

    checkAlgo(algorithm)
    id <- which( zeroSumAlgos[,1] == algorithm)
    dataObject$algorithm <- as.integer( zeroSumAlgos[id,2] )


    if( type %in% zeroSumTypes[ c(3,4,7,8,11,12,15,16),1] )
    {
        dataObject$algorithm   <- as.integer(3)
    }

    dataObject$useApprox   <- checkInteger(useApprox)

    if( dataObject$type <= 4  )
    {
        dataObject$useApprox <- as.integer(FALSE)

    }else if( dataObject$algorithm==1 )
    {
        dataObject$useApprox <- as.integer(TRUE)
    }

    dataObject$diagonalMoves <- checkInteger(diagonalMoves)
    dataObject$polish        <- checkInteger(polish)

    return(dataObject)
}
