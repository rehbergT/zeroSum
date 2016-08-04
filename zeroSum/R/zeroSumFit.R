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
#' @param offset determines if an offset should be used in the
#'               model or not (TRUE/FALSE)
#'
#' @param type choose the regression type: elNet, zeroSumElNet
#
#' @param algorithm choose an algorithm:
#'            CD = Coordinate descent (very fast, not so accurate),
#'            CD+LS = Coordinate descent + local search (fast, very accurate),
#'            LS = Local search (slow, accurate),
#'            SA = Simulated annealing (very slow, very accurate)
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
#' zeroSumFit( x, y, 1.5, 1)
#'
#' @export
zeroSumFit <- function( 
                x, 
                y, 
                lambda, 
                alpha, 
                offset=TRUE, 
                type="zeroSumElNet",
                algorithm="CD+LS",
                precision=1e-6, 
                diagonalMoves=TRUE, 
                polish=0,
                verbose=FALSE,
                beta=NULL) 
{    
    # some basic checks for the passed arguments
    checkNumericMatrix(x, 'x')
    
    checkType( type )
    if( type == "elNet" || type == "zeroSumElNet" ){
        checkNumericVector(y, 'y')
    }

    if( nrow(x) != length(y)) 
    {
            stop("number of rows of x does not match length of y!\n")
    }

    checkDouble( alpha, "alpha")
    checkDouble( lambda, "lambda")
    checkAlgo( algorithm, "algorithm")

    
    if(is.null(colnames(x)))
    {
        tmp <-c("Intercept", seq(1, ncol(x)))
    }else
    {
        tmp <-c("Intercept", colnames(x) )
    }
    
    x <- cbind( rep( 1.0, nrow(x)), x) 
    colnames(x) <- tmp

    N <- nrow(x)
    P <- ncol(x)

    if( is.null(beta) )
        beta <- rep( 0.0, ncol(x) )        
      

    energy1 <- 0
    if( type=="elNet" || type=="zeroSumElNet" )
    {
        energy1 <- vectorElNetCostFunction(x ,y ,beta, lambda, alpha)
    }

    if(verbose)
    { 
        print( sprintf( "Energy before: %e", energy1$cost))
    }
    
    zeroSumRegression( x, y ,beta, lambda, alpha, offset,
                type, algorithm, verbose=FALSE, precision, diagonalMoves,
                polish)
   

    energy2 <- 0
    if( type=="elNet" || type=="zeroSumElNet" )
    {
        energy2 <- vectorElNetCostFunction(x ,y ,beta, lambda, alpha)
    }

    if(verbose)
    { 
        print( sprintf( "Energy later %e   Dif %e  (RSS: %e)", energy2$cost, energy2$cost-energy1$cost,  energy2$rss))
    }
    
    names(beta) <- colnames(x)

    fitresult <- zeroSumFitObject(  lambda, 
                                    alpha, 
                                    beta, 
                                    type, 
                                    algorithm,
                                    diagonalMoves)

    return(fitresult)

}
    
