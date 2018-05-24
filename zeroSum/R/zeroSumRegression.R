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
zeroSumRegression <- function( data )
{
    .Call( "CV", data, PACKAGE="zeroSum")
}
