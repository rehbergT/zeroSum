#' Description of zeroSumFit function
#'
#' Creates a linear model with the elastic-net regularization and the zero-sum
#'      constraint for given x, y, lambda and alpha.
#'
#' @param x data as a numeric matrix object (rows=samples).
#'          The zero-sum regression requires data on the log scale, i.e.
#'          x should be log-transformed data.
#'
#' @param y response vector to be predicted by x (length(y)==nrow(x))
#'
#' @param lambda penalizing parameter of the elastic-net regularization
#'
#' @param alpha Lasso/Ridge adjustment: For alpha = 0 the elastic net becomes
#'              a ridge regularization, for alpha = 1 the elastic net becomes
#'              the lasso regularization
#'
#' @param weights sample weights (must be greater than zero)
#'
#' @param penalty.factor weights for the elatic net regularization
#'        (must be greater than or equal to zero)
#'
#' @param standardize standardize the data (be careful standardization causes that
#'        the scale of the data affects the coefficients! This can act contray to
#'        scale invariance caused by the zero-sum constraint!)
#'
#' @param useOffset determines if an offset should be used in the
#'               model or not (TRUE/FALSE)
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
#' @param precision stopping criterion of the used algorithms. Determines how
#'                  small the improvement of the cost function has to be to stop
#'                  the algorithm. Default is 1e-6.
#'
#' @param diagonalMoves allows the CD to use diagonal moves (can in rare cases
#'                      slightly increase the accuracy of the models but
#'                      increases the computing time)
#'
#' @param polish enables a local search at the end of CD to polish the result
#'               (removes small numeric uncertainties causing very small but non-zero
#'               coeffiencts and has only minimal effects on the computing time)
#'
#' @param beta start coeffiencts of the algorithm
#'
#' @param ... can be used for adjusting internal parameters
#'
#' @return zeroSumFitObject
#'
#' @importFrom methods hasArg
#'
#' @examples
#' set.seed(1)
#' x <- log2(exampleData$x+1)
#' y <- exampleData$y
#' zeroSumFit( x, y, lambda=1.5 )
#'
#' @export
zeroSumFit <- function(
                x,
                y,
                lambda,
                alpha          = 1.0,
                weights        = NULL,
                penalty.factor = NULL,
                standardize    = FALSE,
                useOffset      = TRUE,
                type           = "gaussianZS",
                precision      = 1e-8,
                diagonalMoves  = FALSE,
                polish         = TRUE,
                beta           = NULL,
                ... )
{
    args <- list(...)
    if(hasArg(fusion))     { fusion = args$fusion }         else { fusion <- NULL }
    if(hasArg(gamma))      { gamma = args$gamma }           else { gamma <- 0.0 }
    if(hasArg(useApprox))  { useApprox = args$useApprox }   else { useApprox <- TRUE }
    if(hasArg(downScaler)) { downScaler = args$downScaler } else { downScaler <- 1.0 }
    if(hasArg(algorithm))  { algorithm = args$algorithm }   else { algorithm <- "CD" }
    if(hasArg(zeroSumWeights))  { zeroSumWeights = args$zeroSumWeights }   else { zeroSumWeights <- NULL }

    data <- regressionObject(x, y, beta , lambda, alpha, gamma, type, weights,
                penalty.factor, zeroSumWeights, fusion, precision, useOffset,
                useApprox, downScaler, algorithm, diagonalMoves, polish,
                standardize, nFold=0)

    energy1 <- costFunction( data )

    start <- Sys.time()
    data$result <- zeroSumRegression( data )

    end <- Sys.time()
    runtime <- as.numeric(end-start, units="secs")

    fitresult <- zeroSumCVFitObject( data )
    fitresult$runtime <- runtime

    return(fitresult)

}
