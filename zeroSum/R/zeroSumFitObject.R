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
    xSD <- regressionObject$xSD
    xM  <- regressionObject$xM

    ySD <- regressionObject$ySD
    yM  <- regressionObject$yM

    if(regressionObject$standardize==TRUE)
    {
        if( regressionObject$type %in% zeroSumTypes[ 1:6,2] )
        {
            regressionObject$beta[1] <- ( regressionObject$beta[1] -
                as.numeric( regressionObject$beta[-1] %*% ( xM / xSD )) ) * ySD + yM

            regressionObject$beta[-1] <- regressionObject$beta[-1] / xSD * ySD

        } else if( regressionObject$type %in% zeroSumTypes[ 7:12,2] )
        {
            regressionObject$beta[1] <- ( regressionObject$beta[1] -
                as.numeric( regressionObject$beta[-1] %*% ( xM / xSD )) )

            regressionObject$beta[-1] <- regressionObject$beta[-1] / xSD

        } else if( regressionObject$type %in% zeroSumTypes[ 13:18,2] )
        {
            regressionObject$beta[1,] <- ( regressionObject$beta[1,] -
                as.numeric( t(regressionObject$beta[-1,]) %*% ( xM / xSD ) ) )

            regressionObject$beta[-1,] <- regressionObject$beta[-1,] / xSD
        }
    }
    
    regressionObject$x <- NULL
    regressionObject$y <- NULL
    regressionObject$fusion <- NULL
    
    class(regressionObject) <- append( class(regressionObject),"ZeroSumFit")
    return(regressionObject)

}
