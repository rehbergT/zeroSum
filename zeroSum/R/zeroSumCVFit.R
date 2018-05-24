#' Description of zeroSumCVFit function
#'
#' This function determines for a given dataset x, y and elastic net parameter
#' alpha an optimal lambda value by cross validation (cv). It returns
#' a linear model for the optimal lambda. An appropiate lambda sequence
#' is estimated or can be passed as an argument. For each lambda value
#' an nFold cv error is calculated. The optimal lambda corresponds to the lowest
#' cv error.
#'
#' @param x data as a numeric matrix object (rows=samples).
#'          The zero-sum regression requires data on the log scale, i.e.
#'          x should be log-transformed data.
#'
#' @param y response vector to be predicted by x (length(y)==nrow(x))
#'
#' @param lambda sequence of lambda values to be tested.
#'          If lambda==0 a sequence will be approximated
#'
#' @param lambdaSteps this parameters determines the number of lambda steps between
#'                    lambdaMin and lambdaMax, i.e higher values for lambdaSteps
#'                    increase the resolution of the regularization path.
#'
#' @param alpha Lasso/Ridge adjustment: For alpha = 0 the elastic net becomes
#'              a ridge regularization, for alpha = 1 the elastic net becomes
#'              the lasso regularization
#'
#' @param weights samples weights (must be greater than zero)
#'
#' @param penalty.factor weights for the elatic net regularization
#'        (must be greater than or equal to zero)
#'
#' @param standardize standardize the data (be careful standardization causes that
#'        the scale of the data affects the coefficients! This can act contray to
#'        scale invariance caused by the zero-sum constraint!)
#'
#' @param epsilon If a lambda sequence is estimated, lambdaMax is chosen such
#'              that all coefficients become zero, i.e. lambdaMax is the upper
#'              bound of the lambda sequence. The lower bound is calculated by
#'              lambdaMin = lambdaMax * epsilon and can be adjusted by this
#'              parameter.
#'
#' @param nFold the number of folds used for the cross validation
#'
#' @param foldid allows to determine the folds used for cross validation.
#'
#' @param useOffset determines if an offset should be used in the
#'               model or not (TRUE/FALSE)
#'
#' @param cores The cross validation can be executed in parallel. cores
#'              defines the amount of cpu cores to be used!
#'
#' @param verbose verbose = TRUE enables additional output about the regression
#'
#' @param type choose the regression type:
#'              \describe{
#'                      \item{gaussian:}{}
#'                      \item{gaussianZS:}{}
#'                      \item{binomial:}{}
#'                      \item{binomialZS:}{}
#'                      \item{multinomial:}{}
#'                      \item{multinomialZS:}{}
#'              }
#'
#' @param precision stopping criterion of the used algorithms.
#'                    Determines how small the improvement of the cost function
#'                    has to be to stop the algorithm. Default is 1e-8.
#'
#' @param diagonalMoves allows the CD to use diagonal moves (can in rare cases
#'                      slightly increase the accuracy of the models but
#'                      increases the computing time)
#'
#' @param polish enables a local search at the end of CD to polish the result
#'               (removes small numeric uncertainties causing very small but non-zero
#'               coeffiencts and has only minimal effects on the computing time)
#'
#' @param cvStop    stops the CV progress if the model's CV-error becomes worse
#'                  for lower lambda values. The number of worse
#'                  values which is tolerated is calculated by multiplying
#'                  the lambdaSteps (Default: 100) with the value of cvStop
#'                  parameter. (Default: 0.1). Use cvStop = 0 or FALSE to
#'                  deactivate the devianceStop.
#'
#' @param ... can be used for adjusting internal parameters
#'
#' @importFrom methods hasArg
#'
#' @return zeroSumCVFitObject
#'
#' @examples
#' set.seed(1)
#' x <- log2(exampleData$x+1)
#' y <- exampleData$y
#' fit <- zeroSumCVFit( x, y, alpha=1)
#' plot( fit, "test")
#' coef(fit, s="lambda.min")
#'
#' @export
zeroSumCVFit <- function(
            x,
            y,
            lambda              = 0,
            lambdaSteps         = 100,
            alpha               = 1.0,
            weights             = NULL,
            penalty.factor      = NULL,
            standardize         = FALSE,
            epsilon             = NULL,
            nFold               = 10,
            foldid              = NULL,
            useOffset           = TRUE,
            cores               = 1,
            verbose             = FALSE,
            type                = "gaussianZS",
            precision           = 1e-8,
            diagonalMoves       = FALSE,
            polish              = TRUE,
            cvStop              = 0.1,
            ... )
{
    args <- list(...)
    if(hasArg(fusion))     { fusion = args$fusion }         else { fusion <- NULL }
    if(hasArg(gamma))      { gamma = args$gamma }           else { gamma <- 0.0 }
    if(hasArg(gammaSteps)) { gammaSteps = args$gammaSteps } else { gammaSteps <- 1 }
    if(hasArg(useApprox))  { useApprox = args$useApprox }   else { useApprox <- TRUE }
    if(hasArg(downScaler)) { downScaler = args$downScaler } else { downScaler <- 1.0 }
    if(hasArg(algorithm))  { algorithm = args$algorithm }   else { algorithm <- "CD" }
    if(hasArg(zeroSumWeights))  { zeroSumWeights = args$zeroSumWeights }   else { zeroSumWeights <- NULL }

    data <- regressionObject(x, y, NULL, lambda, alpha, gamma, type, weights,
                penalty.factor, zeroSumWeights, fusion, precision, useOffset,
                useApprox, downScaler, algorithm, diagonalMoves, polish,
                standardize, lambdaSteps, gammaSteps, nFold, foldid, epsilon,
                cvStop, verbose, cores)

    start <- Sys.time()

    data$result <- zeroSumRegression( data )

    end <- Sys.time()
    runtime <- as.numeric(end-start, units="secs")

    if(verbose) {
        print( sprintf("runtime: %.3fs", runtime ))
    }

    fitresult <- zeroSumCVFitObject( data )
    fitresult$runtime <- runtime
    return(fitresult)
}
