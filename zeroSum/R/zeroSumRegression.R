#' Description of zeroSumRegression_C Wrapper function
#'
#' This is a wrapper function for the C regression functions
#'
#' @return regression coefficients are returned
#'
#' @useDynLib zeroSum
#'
#' @keywords internal
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
