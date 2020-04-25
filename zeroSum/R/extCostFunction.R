#' Description of costFunction function
#' wraps input into a regression object and
#' calls the costFunction and returns the result
#'
#' @keywords internal
#'
#' @export
extCostFunction <- function(x,
                            y,
                            beta,
                            alpha = 1.0,
                            lambda = NULL,
                            family = "gaussian",
                            lambdaSteps = 100,
                            weights = NULL,
                            penalty.factor = NULL,
                            zeroSum.weights = NULL,
                            nFold = NULL,
                            foldid = NULL,
                            epsilon = NULL,
                            standardize = FALSE,
                            intercept = TRUE,
                            zeroSum = FALSE,
                            threads = "auto",
                            cvStop = 0.1,
                            useC = FALSE,
                            ...) {
    data <- regressionObject(
        x, y, family, alpha, lambda, lambdaSteps, weights,
        penalty.factor, zeroSum.weights, nFold, foldid, epsilon,
        standardize, intercept, zeroSum, threads, cvStop,
        beta = beta, ...
    )

    return(costFunction(data, useC))
}