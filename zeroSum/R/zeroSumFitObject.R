#' Description of zeroSumFitObject function
#'
#'
#' Creates a zeroSumFitObject which stores all arguments and results of
#' the zeroSumFit() function. 
#'
#' @return zeroSumFitObject
#'
#' @keywords internal
#'
zeroSumFitObject <- function(   lambda, 
                                alpha, 
                                coef,
                                type,
                                algorithm )
{
    zeroSumFit <- list()
    
    zeroSumFit$lambda <- lambda 
    zeroSumFit$alpha <- alpha 

    zeroSumFit$coef <- coef
    zeroSumFit$type <- type
    zeroSumFit$algorithm <- algorithm
    
    class(zeroSumFit) <- append( class(zeroSumFit),"ZeroSumFit")
    return(zeroSumFit) 

}


