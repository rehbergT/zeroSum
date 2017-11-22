#' Description of costFunction function
#' wraps input into a regression object and
#' calls the costFunction and returns the result
#'
#' @keywords internal
#'
#' @export
extCostFunction <- function( x, y, beta, lambda, alpha=1, gamma=0,
        type=zeroSumTypes[1,1], weights=NULL, penalty.factor=NULL, fusion=NULL )
{

    data <- regressionObject( x, y, beta, lambda, alpha, gamma,
                    type=type, weights=weights, penalty.factor=penalty.factor,
                    fusion=fusion, standardize=FALSE, useOffset=FALSE)
    
    return(costFunction(data))
}
