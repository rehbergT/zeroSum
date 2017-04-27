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
#' @param gamma sequence of gamma values to be tested.
#'          If gamma==0 and a fused/fusion regression type set a sequence will be approximated (not implemented yet...)
#'
#' @param gammaSteps this parameters determines the number of gamma steps between
#'                    gammaMin and gammaMax, i.e higher values for gammaSteps
#'                    increase the resolution of the regularization path.
#'
#' @param alpha Lasso/Ridge adjustment: For alpha = 0 the elastic net becomes
#'              a ridge regularization, for alpha = 1 the elastic net becomes
#'              the lasso regularization
#'
#' @param weights samples weights
#'
#' @param penalty.factor weights for the elatic net regularization
#'
#' @param zeroSumWeights weights of the zeroSum constraint
#'
#' @param cSum constant c of the zeroSum constraint. Default 0. Anything else is experimental.
#'
#' @param standardize standardize x and y
#'
#' @param fusion penalizing matrix of the fusion term
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
#' @param useApprox determines if the quadratic approximation of the
#'               log-likelihood or the log-likelihood itself should be used
#'               by the local search algorithm for fitting binomial or
#'               multinomial models
#'
#' @param downScaler allows to reduce the number of moves
#'
#' @param cores The cross validation can be executed in parallel. cores
#'              defines the amount of cpu cores to be used!
#'
#' @param verbose verbose = TRUE enables output
#'
#' @param type choose the regression type:
#'              \describe{
#'                      \item{gaussian:}{linear regression}
#'                      \item{gaussianZS:}{linear zero-Sum regression (default)}
#'                      \item{binomial:}{logistic regression}
#'                      \item{binomialZS:}{logistic zero-Sum regression}
#'              }
#'
#' @param algorithm determines the used algorithm:
#'            \describe{
#'            \item{CD:}{ Coordinate descent (very fast, not so accurate)}
#'            \item{CD+LS:}{ Coordinate descent + Local search (fast, very accurate)}
#'            \item{LS:}{ Local search (slow, accurate)} }
#'
#' @param precision stopping criterion of the used algorithms.
#'                    Determines how small the improvement of the cost function
#'                    has to be to stop the algorithm. Default is 1e-8.
#'
#' @param diagonalMoves allows the CD to use diagonal moves
#'
#' @param lambdaScaler allows to adjust the approximated lambdaMax
#'
#' @param polish enables a local search at the end of CD to polish the result
#'
#' @param cvStop    stops the CV progress if the model's CV-error becomes worse
#'                  for lower lambda values. The number of worse
#'                  values which is tolerated is calculated by multiplying
#'                  the lambdaSteps (Default: 100) with the value of cvStop
#'                  parameter. (Default: 0.1). Use cvStop = 0 or FALSE to
#'                  deactivate the devianceStop.
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
#' @import foreach
#' @importFrom stats rnorm sd
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
            zeroSumWeights      = NULL,
            cSum                = 0.0,
            standardize         = FALSE,
            gamma               = 0.0,
            gammaSteps          = 1,
            fusion              = NULL,
            epsilon             = 0.001,
            nFold               = 10,
            foldid              = NULL,
            useOffset           = TRUE,
            useApprox           = TRUE,
            downScaler          = 1,
            cores               = 1,
            verbose             = FALSE,
            type                = "gaussianZS",
            algorithm           = "CD",
            precision           = 1e-8,
            diagonalMoves       = TRUE,
            lambdaScaler        = 1,
            polish              = 0,
            cvStop              = 0.1 )
{
    # some basic checks for the passed arguments
    data <- regressionObject(x, y, NULL, lambda, alpha,
                gamma, cSum, type, weights, zeroSumWeights,
                penalty.factor, fusion, precision,
                useOffset, useApprox, downScaler, algorithm,
                diagonalMoves, polish, standardize, lambdaSteps,
                gammaSteps, nFold, foldid, epsilon, cvStop, verbose,
                lambdaScaler, cores)

    if( !(data$type %in% zeroSumTypes[c(1,2,7,8),2] ) )
    {
        print("Experimental data type!")
        print("This is work in progress so take the results with a grain of salt!")
    }

    if(verbose) start <- Sys.time()

    data$result <- zeroSumRegression( data, TRUE )

    if(verbose) {
        end <- Sys.time()
        print( sprintf("runtime: %.3fs", as.numeric(end-start, units="secs")))
    }

    fitresult <- zeroSumCVFitObject( data )
    return(fitresult)
}
