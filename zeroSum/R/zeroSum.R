#' Description of zeroSum function
#'
#' This function fits a zero-sum model for a given dataset x, y. Gaussian,
#' binomial logistic, mulitnomial logistic and cox proportional hazard
#' regression are supported. The models are regularization using the
#' elastic-net the optimal regularization is determined by cross validation.
#'
#' @param x data as a numeric matrix object (rows=samples). The zero-sum
#'        regression requires data on the log scale, i.e. x should be
#'        log-transformed data.
#'
#' @param y response vector to be predicted by x (length(y)==nrow(x))
#'
#' @param family choose the regression type:
#'        \describe{
#'            \item{gaussian:}{numeric response}
#'            \item{binomial:}{}
#'            \item{multinomial:}{}
#'            \item{cox:}{y should be matrix with two columns, the first must
#'                        contain the event time and the second indicating the
#'                         type: 1 = event has occured, 0 = right censoring}
#'        }
#'
#' @param alpha Lasso/Ridge adjustment: For alpha = 0 the elastic net
#'       corresponds to the ridge regularization, for alpha = 1 the elastic net
#'       corresponds to lasso regularization
#'
#' @param lambda sequence of lambda values to be tested by cross validation. If
#'        only one value is supplied only a regression for that specific lambda
#'        is performed. If more values are supplied a cross validation is
#'        performed and the optimal lambda (lambdaMin) determined. If no lambda
#'        is supplied a suitable sequence is automatically determined.
#'
#' @param lambdaSteps this parameters defines the number of lambda steps
#'        between lambdaMin and lambdaMax, i.e higher values for lambdaSteps
#'        increase the resolution of the regularization path.
#'
#' @param weights samples weights vector of length nrow(x). Can for example be
#'        used to mitigate unbalanced group sizes in the logistic regression
#'        case. (must be greater than zero)
#'
#' @param penalty.factor weights vector for the elatic net regularization of
#'        length ncol(x). By setting a weight to 0 the corresponding feature
#'        will not be regularized and thus will be part of the resulting model.
#'        (must be greater than or equal to zero, default 1).
#
#' @param zeroSum.weights weights vector for the zero-sum constraint of
#'        length ncol(x). By setting a weight to 0 the corresponding feature
#'        will be excluded from the zero-sum constraint.
#'        (must be greater than or equal to zero, default 1)
#'
#' @param standardize standardize the data. Be careful! Standardization causes
#'        that the scale of the data affects the coefficients! This acts
#'        contray to scale invariance caused by the zero-sum constraint!.
#'        (Default false)
#'
#' @param epsilon If a lambda sequence is estimated, lambdaMax is chosen such
#'        that all coefficients become zero, i.e. lambdaMax is the upper bound
#'        of the lambda sequence. The lower bound is calculated by
#'        lambdaMin = lambdaMax * epsilon. Thus epsilon can be used to adjust
#'        this range.
#'
#' @param nFold the number of folds used by the cross validation (Default 10)
#'
#' @param foldid defines the folds used by the cross validation. For example a
#         vector c(1,1,2,2,3,3,...) define that sample the frist two samples
#'        are in fold 1, the third and fourth sample are in fold 2, ...
#'
#' @param intercept determines if an intercept should be used in the
#'        model or not (TRUE/FALSE)
#'
#' @param zeroSum determines if the zero-sum constraint should be used. If set
#'        to false the same results as with the glmnet package should be
#'        obtained. Note that the standardize is default TRUE in the glmnet
#'        package. default TRUE.
#'
#' @param threads Threads defines the number of cpu threads used to parallelize
#'        the cross validation. Thus a parallel speed-up is only achieveable up
#'        to nFold+1 threads. The argument "auto" (Default) can be used to
#'        automatically select the maximum number of parallel executable
#'        threads.
#'
#' @param cvStop stops the CV progress if the model's CV-error becomes worse
#'        for lower lambda values. The number of worse
#'        values which is tolerated is calculated by multiplying
#'        the lambdaSteps (Default: 100) with the value of cvStop
#'        parameter. (Default: 0.1). Use cvStop = 0 or FALSE to
#'        deactivate the devianceStop.
#'
#' @param ... can be used for adjusting internal parameters
#'
#' @import methods
#'
#' @return zeroSumFit object
#'
#' @examples
#' set.seed(1)
#' x <- log2(exampleData$x+1)
#' y <- exampleData$y
#' fit <- zeroSum( x, y, alpha=1)
#' plot( fit, "test")
#' coef(fit, s="lambda.min")
#'
#'
#' @useDynLib zeroSum, .registration = TRUE
#' @export
zeroSum <- function(
            x,
            y,
            family              = "gaussian",
            alpha               = 1.0,
            lambda              = NULL,
            lambdaSteps         = 100,
            weights             = NULL,
            penalty.factor      = NULL,
            zeroSum.weights     = NULL,
            nFold               = NULL,
            foldid              = NULL,
            epsilon             = NULL,
            standardize         = FALSE,
            intercept           = TRUE,
            zeroSum             = TRUE,
            threads             = "auto",
            cvStop              = 0.1,
            ...) {

    args <- list(...)

    if (methods::hasArg("beta")) {
         beta <- args$beta
    } else {
         beta <- NULL
    }

    if (methods::hasArg("center")) {
        center <- args$center
    } else {
        center <- TRUE
    }

    if (methods::hasArg("fusion")) {
         fusion <- args$fusion
    } else {
        fusion <- NULL
    }

    if (methods::hasArg("gamma")) {
        gamma <- args$gamma
    } else {
        gamma <- 0.0
    }

    if (methods::hasArg("gammaSteps")) {
        gammaSteps <- args$gammaSteps
    } else {
        gammaSteps <- 1
    }

    if (methods::hasArg("useApprox")) {
        useApprox <- args$useApprox
    } else {
        useApprox <- TRUE
    }

    if (methods::hasArg("downScaler")) {
        downScaler <- args$downScaler
    } else {
        downScaler <- 1.0
    }

    if (methods::hasArg("algorithm")) {
        algorithm <- args$algorithm
    } else {
        algorithm <- "CD"
    }

    if (methods::hasArg("verbose")) {
        verbose <- args$verbose
    } else {
        verbose <- FALSE
    }

    if (methods::hasArg("polish")) {
        usePolish <- args$polish
    } else {
        usePolish <- TRUE
    }

    if (methods::hasArg("precision")) {
        precision <- args$precision
    } else {
        precision <- 1e-8
    }

    if (methods::hasArg("rotatedUpdates")) {
        rotatedUpdates <- args$rotatedUpdates
    } else {
        rotatedUpdates <- FALSE
    }

    data <- regressionObject(x, y, beta, alpha, lambda, gamma, family, weights,
                penalty.factor, zeroSum.weights, fusion, precision, intercept,
                useApprox, downScaler, algorithm, rotatedUpdates, usePolish,
                standardize, lambdaSteps, gammaSteps, nFold, foldid, epsilon,
                cvStop, verbose, threads, center, zeroSum)

    start <- Sys.time()

    data$result <- .Call("CV", data, PACKAGE = "zeroSum")

    end <- Sys.time()
    runtime <- as.numeric(end - start, units = "secs")

    if (verbose) {
        print(sprintf("runtime: %.3fs", runtime))
    }

    fitresult <- zeroSumObject(data)
    fitresult$runtime <- runtime
    return(fitresult)
}
