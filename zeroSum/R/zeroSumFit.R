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
#' @param weights samples weights (must be greater than zero)
#'
#' @param penalty.factor weights for the elatic net regularization
#'        (must be greater than or equal to zero)
#'
#' @param zeroSumWeights weights of the zeroSum constraint
#'        (must be greater than zero)
#'
#' @param cSum constant c of the zeroSum constraint
#'
#' @param standardize standardize x and y
#'
#' @param gamma penalizing parameter of the fusion term
#'
#' @param fusion penalizing matrix of the fusion term
#'
#' @param useOffset determines if an offset should be used in the
#'               model or not (TRUE/FALSE)
#'
#' @param downScaler allows to reduce the number of moves
#'
#' @param useApprox determines if the quadratic approximation of the
#'               log-likelihood or the log-likelihood itself should be used
#'               by the local search algorithm for fitting binomial or
#'               multinomial models
#'
#' @param type choose the regression type:
#'              \describe{
#'                      \item{gaussian:}{}
#'                      \item{gaussianZS:}{}
#'                      \item{fusionGaussian:}{}
#'                      \item{fusionGaussianZS:}{}
#'                      \item{binomial:}{}
#'                      \item{binomialZS:}{}
#'                      \item{fusionBinomial:}{}
#'                      \item{fusionBinomialZS:}{}
#'                      \item{multinomial:}{}
#'                      \item{multinomialZS:}{}
#'                      \item{fusionMultinomial:}{}
#'                      \item{fusionMultinomialZS:}{}
#'              }
#'
#' @param algorithm choose an algorithm:
#'            \describe{
#'            \item{CD:}{ Coordinate descent (very fast, not so accurate)}
#'            \item{CD+LS:}{ Coordinate descent + Local search (fast, very accurate)}
#'            \item{LS:}{ Local search (slow, accurate)} }
#'
#'
#' @param precision stopping criterion of the used algorithms. Determines how
#'                  small the improvement of the cost function has to be to stop
#'                  the algorithm. Default is 1e-6.
#'
#' @param diagonalMoves allows the coordinate descent to use diagonal moves
#'
#' @param polish enables a local search at the end of CD to polish the result
#'
#' @param verbose verbose = TRUE enables output
#'
#' @param beta start coeffiencts of algorithm
#'
#'
#' @return zeroSumFitObject
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
                zeroSumWeights = NULL,
                cSum           = 0.0,
                standardize    = FALSE,
                gamma          = 0.0,
                fusion         = NULL,
                useOffset      = TRUE,
                useApprox      = TRUE,
                downScaler     = 1,
                type           = "gaussianZS",
                algorithm      = "CD",
                precision      = 1e-8,
                diagonalMoves  = TRUE,
                polish         = 0,
                verbose        = FALSE,
                beta           = NULL )
{
    data <- regressionObject(x, y, beta , lambda, alpha, gamma, cSum,
        type, weights, zeroSumWeights, penalty.factor, fusion,
        precision, useOffset, useApprox, downScaler,
        algorithm, diagonalMoves, polish, standardize)

    energy1 <- costFunction( data )
    if(verbose)
    {
        print( sprintf( "Energy before: %e", energy1$cost))
        start <- Sys.time()
    }

    zeroSumRegression( data, FALSE )

    if(verbose) end <- Sys.time()

    energy2 <- costFunction( data )

    if(verbose)
    {
        print( sprintf( "Energy later %e   Dif %e runtime: %.3fs",
                energy2$cost, energy2$cost-energy1$cost,
                as.numeric(end-start, units="secs") ))
    }

    fitresult <- zeroSumFitObject( data )

    return(fitresult)

}
