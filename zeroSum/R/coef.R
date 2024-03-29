#' Description of the coef function for zeroSum s3 objects
#'
#' This function returns the coefficients of a zeroSum object object.
#'
#' @param object fit object generated by zeroSum()
#'
#' @param s determines which lambda of a zeroSumCVFit should be returned:
#'          lambda.min or lamda.1SE
#'
#' @param ... other arguments for the normal predict function if the fit
#'            is not a zeroSumFit object
#'
#' @return estimated coefficients
#'
#' @examples
#' set.seed(1)
#' x <- log2(exampleData$x + 1)
#' y <- exampleData$y
#' fit <- zeroSum(x, y, alpha = 1)
#' coef(fit, s = "lambda.min")
#' @importFrom stats coef
#'
#' @export
coef.zeroSum <- function(object = NULL, s = "lambda.min", ...) {
    if (s == "lambda.min") {
        beta <- object$coef[[object$lambdaMinIndex]]
    } else if (s == "lambda.1SE" || s == "lambda.1se") {
        beta <- object$coef[[object$lambda1SEIndex]]
    } else {
        beta <- object$coef[[s]]
    }
    rownames(beta) <- object$variables.names

    return(beta)
}
