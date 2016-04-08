#' Description of zeroSumRegression_C Wrapper function
#'
#' This is a wrapper function for the C regression functions
#'
#' @param x data as a numeric matrix object (rows=samples). 
#'          The zero-sum regression requires data on the log scale, i.e.
#'          x should be log-transformed data
#'
#' @param y response vector to be predicted by x (length(y)==nrow(x))
#'
#' @param beta coefficients used as warm start for optimization
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
#' @param algorithm 
#'                    0 = coordinate descent 
#'                    1 = simulated annealing
#'                    2 = local search 
#'                    3 = coordinate descent + local search
#'
#' @param verbose complete output of each gradient calculation
#'
#' @param precision stopping criterion of the used algorithms. Determines how
#'                  small the improvement of the cost function has to be to stop
#'                  the algorithm.
#'
#' @param diagonalMoves allows the coordinate descent to use diagonal moves 
#'
#' @param polish enables a local search at the end of CD to polish the result
#'
#' @return regression coefficients are returned
#'
#' @keywords internal
#'
#' @useDynLib zeroSum
#'
zeroSumRegression <- function( x, y, beta, lambda, alpha, offset, type, 
        algorithm, verbose, precision, diagonalMoves, polish) 
{
        
    typeAsInt <- 0
    if( type == "elNet")
    {
        typeAsInt <- 0
    } else if( type == "zeroSumElNet" )
    {
        typeAsInt <- 1
    }
    
    algoAsInt <- 0
    if( algorithm == "CD")
    {
        algoAsInt <- 0
    } else if( algorithm == "SA" )
    {
        algoAsInt <- 1
    } else if( algorithm == "LS")
    {
        algoAsInt <- 2
    } else if( algorithm == "CD+LS")
    {
        algoAsInt <- 3
    }
     
       
    .Call( "CallWrapper", x , y , beta, lambda, alpha,
        as.integer(offset), as.integer(typeAsInt),
        as.integer(algoAsInt), precision, as.integer(diagonalMoves), 
        as.integer(polish), PACKAGE="zeroSum")
     
    ## try to remove numerical uncertainties
    if( ( type == "zeroSumElNet" || type == "zeroSumLogistic" ) && 
        sum(beta[-1] != 0.0 ) )
    {
        delta = sum(beta[-1])
            
        ## get first non zero coefs and add the small value
        ids <- which( beta[-1] != 0.0 )
        if( length(ids) > 0){
            beta[ ids[1]+1 ] <- beta[ ids[1]+1 ] - delta
        }            
    }        

}
