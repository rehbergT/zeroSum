#' Description of costFunction function
#' wraps input into a regression object and
#' calls the costFunction and returns the result
#'
#' @keywords internal
#'
#' @export
extCostFunction <- function(x, y, beta , lambda, alpha, gamma=0, cSum=0,
        type=zeroSumTypes[1,1], weights=NULL, zeroSumWeights=NULL, 
        penalty.factor=NULL, fusion=NULL, standardize=FALSE )
{
    data <- regressionObject(x, y, beta , lambda, alpha, gamma,
        type=type, weights=weights, zeroSumWeights=zeroSumWeights, 
        penalty.factor=penalty.factor, fusion=fusion, standardize=standardize )
        
    return(costFunction(data))
}