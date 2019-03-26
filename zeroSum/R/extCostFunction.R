#' Description of costFunction function
#' wraps input into a regression object and
#' calls the costFunction and returns the result
#'
#' @keywords internal
#'
#' @export
extCostFunction <- function(x, y, beta, alpha = 1, lambda = 0, gamma = 0,
                            family = zeroSumTypes[1, 1], weights = NULL,
                            penalty.factor = NULL, fusion = NULL,
                            useC = FALSE) {
    data <- regressionObject(x, y, beta, alpha, lambda, gamma,
        type = family, weights = weights, penalty.factor = penalty.factor,
        fusion = fusion, standardize = FALSE, useIntercept = FALSE, nFold = 0,
        useZeroSum = FALSE, center = FALSE
    )

    return(costFunction(data, useC))
}
