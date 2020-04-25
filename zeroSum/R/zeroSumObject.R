#' Description of zeroSumObject function
#'
#' Creates a zeroSumCVFitObject which stores all arguments and results of
#' the zeroSumCVFit() function.
#'
#' @return zeroSumCVFitObject
#'
#' @keywords internal
#'
zeroSumObject <- function(obj) {
    K <- obj$K
    P <- obj$P
    N <- obj$N

    tmp <- matrix(obj$result, ncol = (5 + K * (P + 1) + N * K), byrow = TRUE)
    if (nrow(tmp) == 0) {
        stop("Detected empty result! Returning NULL")
        return(NULL)
    }

    cv_predict <- tmp[, -c(1:(5 + K * (P + 1))), drop = FALSE]

    tmp <- tmp[, c(1:(5 + K * (P + 1))), drop = FALSE]
    numberCoef <- rep(0, nrow(tmp))

    obj$coef <- list()
    obj$cv_predict <- list()
    for (i in 1:nrow(tmp)) {
        obj$coef[[i]] <- Matrix::Matrix(tmp[i, -c(1:5)],
            ncol = K,
            sparse = TRUE
        )
        numberCoef[i] <- sum(rowSums(as.matrix(
            abs(obj$coef[[i]][-1, , drop = FALSE])
        )) != 0)
        obj$cv_predict[[i]] <- matrix(cv_predict[i, ], ncol = obj$K)
    }

    if (is.null(colnames(obj$x))) {
        obj$variables.names <- c("intercept", as.character(1:P))
    } else {
        obj$variables.names <- c("intercept", colnames(obj$x))
    }

    obj$cv_stats <- cbind(tmp[, 1:5, drop = FALSE], numberCoef)
    colnames(obj$cv_stats) <- c(
        "gamma", "lambda", "training error",
        "CV error", "cv error sd", "non zero coefficients"
    )

    lambdaMin <- which.min(obj$cv_stats[, 4])
    maxCV_SD <- obj$cv_stats[lambdaMin, 4] + obj$cv_stats[lambdaMin, 5]

    ## find lambda where the CVerror is as close as possible to minCV + SD
    distances <- abs(obj$cv_stats[1:lambdaMin, 4] - maxCV_SD)
    lambda1SE <- which.min(distances)

    obj$lambdaMinIndex <- lambdaMin
    obj$lambda1SEIndex <- lambda1SE

    obj$LambdaMin <- obj$lambda[lambdaMin]
    obj$Lambda1SE <- obj$lambda[lambda1SE]

    obj$x <- NULL
    obj$y <- NULL

    obj$v <- NULL
    obj$u <- NULL
    obj$w <- NULL
    obj$v <- NULL
    obj$downScaler <- NULL
    obj$cSum <- NULL
    obj$lambdaSteps <- NULL

    obj$N <- NULL
    obj$K <- NULL
    obj$M <- NULL

    obj$fusion <- NULL
    obj$result <- NULL
    obj$beta <- NULL

    class(obj) <- append(class(obj), "zeroSum")
    return(obj)
}