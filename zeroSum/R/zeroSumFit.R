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
#'
#' @return zeroSumFitObject
#'
#' @examples
#' set.seed(1)
#' data <- simulateData()
#' zeroSumFit( data$x, data$y, 0.05, 0.5)
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
                polish=10,
                verbose=FALSE) 
{    
    if( class(x) != "matrix" | ( typeof(x) != "double" ) )
    {
            stop("type of passed x is not a numeric matrix\n")
    }  
      
    if( (class(y) != "numeric" | typeof(y) != 'double') &&
        (class(y) != "integer" | typeof(y) != 'integer')  ) 
    {
        stop("type of passed y is not numeric or integer\n")
    }

    if( nrow(x) != length(y)) 
    {
            stop("number of rows of x does not match length of y!\n")
    }

    if( class(lambda) != "numeric" | typeof(lambda) != 'double'   ) 
    {
            stop("type of passed lambda is not numeric\n")
    }

    if( class(alpha) != "numeric" | typeof(alpha) != 'double'   ) 
    {
            stop("type of passed alpha is not numeric\n")
    }  
      
    if( class(algorithm) != "character" & typeof(algorithm) != "character" |
            ( algorithm != "CD"   & 
              algorithm != "SA"   &
              algorithm != "LS"   &
              algorithm != "CD+LS"))  
    {
            cat( "Selected algorithm is not valid\n")
            cat( "Now using CD+LS\n" )
            algorithmAllSamples <- "CD+LS"
    }
    if( class(type) != "character" & typeof(type) != "character" |
        (   type != "elNet" & 
            type != "zeroSumElNet" )) 
    {
            cat( "Selected type is not valid\n")
            cat("Use zeroSumElNet (default) or elNet\n" )
            type <- "zeroSumElNet"
    }
    
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
    beta <- rep( 0.0, ncol(x) )        
      

    energy1 <- 0
    if( type=="elNet" || type=="zeroSumElNet" )
    {
        energy1 <- vectorElNetCostFunction(x ,y ,beta, lambda, alpha)$cost
    }

    if(verbose)
    { 
        print( sprintf( "Energy before: %e", energy1))
    }
    
    zeroSumRegression( x, y ,beta, lambda, alpha, offset,
                type, algorithm, verbose=FALSE, precision, diagonalMoves,
                polish)
   

    energy2 <- 0
    if( type=="elNet" || type=="zeroSumElNet" )
    {
        energy2 <- vectorElNetCostFunction(x ,y ,beta, lambda, alpha)$cost
    }

    if(verbose)
    { 
        print( sprintf( "Energy later %e   Dif %e", energy2, energy2-energy1 ))
    }
    
    names(beta) <- colnames(x)

    fitresult <- zeroSumFitObject(  lambda, 
                                    alpha, 
                                    beta, 
                                    type, 
                                    algorithm )

    return(fitresult)

}
    
