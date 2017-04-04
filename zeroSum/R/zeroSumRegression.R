#' Description of zeroSumRegression_C Wrapper function
#'
#' This is a wrapper function for the C regression functions
#'
#' @return regression coefficients are returned
#'
#' @import Matrix
#'
#' @useDynLib zeroSum
#'
#' @keywords internal
#'
zeroSumRegression <- function( data, CV )
{
    if( CV==TRUE )
    {
        .Call( "CV", data,  PACKAGE="zeroSum")
    } else
    {
        .Call( "CallWrapper", data,  PACKAGE="zeroSum")
    }
}
