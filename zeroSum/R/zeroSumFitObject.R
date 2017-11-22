#' Description of zeroSumFitObject function
#'
#' Creates a zeroSumFitObject which stores all arguments and results of
#' the zeroSumFit() function.
#'
#' @return zeroSumFitObject
#'
#' @keywords internal
#'
zeroSumFitObject <- function( regressionObject )
{
    if( regressionObject$calcOffsetByCentering == TRUE )
    {
        regressionObject$useOffset <- 1
    }

    xSD <- regressionObject$xSD
    xM  <- regressionObject$xM

    ySD <- regressionObject$ySD
    yM  <- regressionObject$yM

    regressionObject$betaOrg <- regressionObject$beta

    if(regressionObject$standardize==TRUE)
    {
        if( regressionObject$useOffset ) regressionObject$beta[1,] <- ( regressionObject$beta[1,] -
             ( xM / xSD ) %*% regressionObject$beta[-1,] ) * ySD + yM
        regressionObject$beta[-1,] <- regressionObject$beta[-1,] / xSD * ySD

    } else if ( regressionObject$useOffset==1 ) {
        regressionObject$beta[1,] <- regressionObject$beta[1,] -
            xM %*% regressionObject$beta[-1,] + yM
    }

    regressionObject$x <- NULL
    regressionObject$y <- NULL
    regressionObject$fusion <- NULL

    class(regressionObject) <- append( class(regressionObject),"ZeroSumFit")
    return(regressionObject)

}
