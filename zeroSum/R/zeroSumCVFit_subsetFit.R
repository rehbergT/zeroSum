#' Description of zeroSumCVFit_subsetFit function
#'
#' Creates an elastic net cross validated regression for x,y given alpha.
#'
#' @param beta coefficients used as warm start for optimization
#'
#' @param x data as a numeric matrix object (rows=samples). For zero-sum
#'          this x has to be log-transformed data!
#'
#' @param y response vector to be predicted by x (length(y)==nrow(x))
#'
#' @param val samples to be left out
#'
#' @param lambda penalizing parameter of the elastic-net regularization
#'
#' @param alpha Lasso/Ridge adjustment: For alpha = 0 the elastic net becomes
#'              a ridge regularization, for alpha = 1 the elastic net becomes
#'              the lasso regularization
#'
#' @param offset determines if an offset should be used in the
#'               model or not (TRUE/FALSE)
#'
#' @param type choose the regression type: elNet, zeroSumElNet
#'
#' @param algorithm determines the used algorithm:
#'            CD = Coordinate descent (very fast, not so accurate),
#'            CD+LS = Coordinate descent + local search (fast, very accurate),
#'            LS = Local search (slow, accurate),
#'            SA = Simulated annealing (very slow, very accurate)
#'
#' @param precision stopping criterion of the used algorithms. Determines how
#'                  small the improvement of the cost function has to be to stop
#'                  the algorithm.
#'
#' @param diagonalMoves allows the coordinate descent to use diagonal moves 
#'
#' @param polish enables a local search at the end of CD to polish the result
#'
#' @return cross validated Fit
#'
#' @keywords internal
#'
zeroSumCVFit_subsetFit <- function( beta, x, y, val, lambda, alpha, offset, type, 
            algorithm, precision, diagonalMoves, polish )
{
    out <- list()

    zeroSumRegression( 
        x[-val,], y[-val], beta, lambda, alpha, offset, type, algorithm,
        verbose=FALSE, precision, diagonalMoves, polish )
        
    predict <- x[val,] %*% beta
    out$mean <- mean( ( y[val] - predict)^2 )

    return(out)
}
