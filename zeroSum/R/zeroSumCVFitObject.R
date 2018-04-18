#' Description of zeroSumCVFitObject function
#'
#' Creates a zeroSumCVFitObject which stores all arguments and results of
#' the zeroSumCVFit() function.
#'
#' @return zeroSumCVFitObject
#'
#' @keywords internal
#'
zeroSumCVFitObject <- function( obj )
{
    K <- obj$K
    P <- obj$P
    N <- obj$N

    tmp <- matrix( obj$result, ncol=(5+K*(P+1)+N*K), byrow=TRUE )

    cv_predict <- tmp[ , -c(1:(5+K*(P+1))), drop=FALSE ]

    tmp <- tmp[ , c(1:(5+K*(P+1))), drop=FALSE ]
    numberCoef <- rep(0,nrow(tmp))

    obj$coef <- list()
    obj$cv_predict <- list()
    for(i in 1:nrow(tmp))
    {
        obj$coef[[i]] <- Matrix( tmp[i,-c(1:5)], ncol=K, sparse=TRUE )
        numberCoef[i] <- sum( rowSums( abs(obj$coef[[i]][-1,,drop=FALSE]) ) != 0 )
        obj$cv_predict[[i]] <- matrix( cv_predict[i,], ncol=obj$K )
    }

    if(is.null(colnames(obj$x))) {
        obj$varNames <- c( "intercept", as.character(1:P))
    } else {
        obj$varNames <- c( "intercept", colnames(obj$x))
    }

    obj$cv_stats <- cbind( tmp[,1:5, drop=FALSE], numberCoef )
    colnames(obj$cv_stats) <- c( "gamma", "lambda", "training error",
                    "CV error", "cv error sd", "non zero coefficients")

    maxLogLikeCV <- max( obj$cv_stats[,4] )
    lambdaMin    <- which( obj$cv_stats[,4, drop=FALSE] == maxLogLikeCV )[1]
    maxCV_SD     <- maxLogLikeCV - obj$cv_stats[lambdaMin,5]

    ## find lambda where the CVerror is as close as possible to minCV + SD
    distances <- abs( obj$cv_stats[1:lambdaMin,4] - maxCV_SD)
    distFrom_minCV_SD <- min( distances )
    lambda1SE <- which( distances == distFrom_minCV_SD )

    obj$LambdaMinIndex <- lambdaMin
    obj$Lambda1SEIndex <- lambda1SE

    obj$LambdaMin <- obj$lambda[lambdaMin]
    obj$Lambda1SE <- obj$lambda[lambda1SE]


    if( obj$type <= 4 )
    {
        for(j in 3:4)
        {
            for(i in 1:nrow(obj$cv_stats))
                obj$cv_stats[i,j] <- obj$cv_stats[i,j] * (-2.0)
        }
        for(i in 1:nrow(obj$cv_stats))
            obj$cv_stats[i,5] <- obj$cv_stats[i,5] * 2.0


        if( obj$standardize == TRUE )
        {
            obj$lambda <- obj$lambda * obj$ySD
            obj$cv_stats[,2] <- obj$cv_stats[,2] * obj$ySD

            for(j in 3:5)
            {
                for(i in 1:nrow(obj$cv_stats))
                    obj$cv_stats[i,j] <- obj$cv_stats[i,j] * obj$ySD^2
            }
        }

    }


    if( obj$calcOffsetByCentering == TRUE )
    {
        obj$useOffset <- 1
    }

    xSD <- obj$xSD
    xM  <- obj$xM

    ySD <- obj$ySD
    yM  <- obj$yM

    ## revert standardization or centering
    for(i in 1:length(obj$coef))
    {
        beta <- obj$coef[[i]]

        if(obj$standardize==TRUE)
        {
            if( obj$useOffset ) beta[1,] <- ( beta[1,] - as.numeric( beta[-1,] %*% ( xM / xSD )) ) * ySD + yM
            beta[-1,] <- beta[-1,] / xSD * ySD

            obj$cv_predict[[i]] <- obj$cv_predict[[i]] * ySD + yM

        } else if( obj$useOffset )
        {
            beta[1,] <- beta[1,] - as.numeric( t(beta[-1,]) %*% xM ) + yM
            obj$cv_predict[[i]] <- obj$cv_predict[[i]] + yM
        }

        obj$coef[[i]] <- beta
    }

    obj$x <- NULL
    obj$y <- NULL

    obj$xM <- NULL
    obj$xSD <- NULL

    obj$yM <- NULL
    obj$ySD <- NULL

    obj$v <- NULL
    obj$u <- NULL
    obj$w <- NULL
    obj$v <- NULL
    obj$downScaler <- NULL
    obj$cSum <- NULL
    obj$lambdaSteps <- NULL
    obj$calcOffsetByCentering <- NULL

    obj$N <- NULL
    obj$K <- NULL
    obj$M <- NULL

    obj$fusion <- NULL
    obj$result <- NULL
    obj$beta   <- NULL

    class(obj) <- append( class(obj), "ZeroSumCVFit")
    return(obj)

}
