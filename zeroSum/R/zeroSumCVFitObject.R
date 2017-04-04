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
    K <- ncol(obj$y)
    P <- ncol(obj$x)
    N <- nrow(obj$x)
    
    tmp <- matrix( obj$result, ncol=(5+K*(P+1)+N*K), byrow=TRUE )
    tmp <- tmp[ rowSums( abs(tmp) ) != 0, , drop=FALSE ]
    
    cv_predict <- tmp[ , -c(1:(5+K*(P+1))), drop=FALSE ]
    
    tmp <- tmp[ , c(1:(5+K*(P+1))), drop=FALSE ]
    
    numberCoef <- rep(0,nrow(tmp))
    if( obj$type <= 12 )
    {
        obj$coef <- tmp[,-c(1:5), drop=FALSE]
        colnames(obj$coef) <- c( "intercept", colnames(obj$x))
        numberCoef <- rowSums( obj$coef[,-1, drop=FALSE]!=0)
        obj$cv_predict <- cv_predict
    }
    else
    {
        obj$coef <- list()
        obj$cv_predict <- list()
        for(i in 1:nrow(tmp))
        {
            obj$coef[[i]] <- matrix( tmp[i,-c(1:5)], ncol=obj$K )
            numberCoef[i] <- sum( rowSums( obj$coef[[i]][-1,,drop=FALSE] ) != 0 )
            
            obj$cv_predict[[i]] <- matrix( cv_predict[i,], ncol=obj$K )
        }
        
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
    
    if(obj$standardize==TRUE)
    {
        xSD <- obj$xSD
        xM  <- obj$xM

        ySD <- obj$ySD
        yM  <- obj$yM
    
        if( obj$type <= 12 )
        {
            for(i in 1:nrow(obj$coef))
            {    
                if( obj$type %in% zeroSumTypes[ 1:6,2] )
                {
                    obj$coef[i,1] <- ( obj$coef[i,1] -
                        as.numeric( obj$coef[i,-1] %*% ( xM / xSD )) ) * ySD + yM

                    obj$coef[i,-1] <- obj$coef[i,-1] / xSD * ySD

                } else if( obj$type %in% zeroSumTypes[ 7:12,2] )
                {
                    obj$coef[i,1] <- ( obj$coef[i,1] -
                        as.numeric( obj$coef[i,-1] %*% ( xM / xSD )) )

                    obj$coef[i,-1] <- obj$coef[i,-1] / xSD

                }
            }
        }
        else
        {
            for(i in 1:length(obj$coef))
            {    
                # todo
            }
        }
    }
    
    obj$x <- NULL
    obj$y <- NULL
    obj$fusion <- NULL
    
    class(obj) <- append( class(obj), "ZeroSumCVFit")
    return(obj)

}
