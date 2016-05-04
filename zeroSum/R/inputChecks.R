#' Description of inputChecks function
#'
#' These function are for checking user input
#'
#' @return true or stop
#'
#' @keywords internal
#'
checkNumericMatrix <- function( x, varName)
{
    if( class(x) != "matrix" | typeof(x) != "double"  ){
        message <- sprintf("Type of %s is not a numeric matrix\n", varName)
        stop(message)
    }
}

checkNumericVector <- function( x, varName)
{
    if( class(x) != "numeric" | typeof(x) != "double"  ){
        message <- sprintf("Type of %s is not a numeric vector\n", varName)
        stop(message)
    }
}

checkBinominalVector <- function( x, varName)
{
    if( class(x) != "integer" | typeof(x) != "integer"  ){
        message <- sprintf("Type of %s is not a integer vector\n", varName)
        stop(message)
    }
}

checkType <- function( type )
{
    if( class(type) != "character" & typeof(type) != "character" |
        (   type != "elNet" &
            type != "zeroSumElNet"))
    {
        message <- paste0( "Selected type is not valid\nUse zeroSumElNet ",
                              "(default) or elNet!\n")
        stop(message)
    }   
}

checkAlgo <- function( algo, name)
{
    if( class(algo) != "character" &  typeof(algo) != "character" 
        | ( algo != "CD" & algo != "SA" & algo != "LS" & algo != "CD+LS"))
    {
        message <- sprintf( "Selected %s is not valid\n")
        stop(message)
    }
}

checkDouble <- function( x, name)
{
    if( class(x) != "numeric" | typeof(x) != "double"   ) 
    {
        message <- sprintf("Type of %s is not numeric\n", x)
        stop(message)
    }
}

checkInteger <- function( x, name)
{
    newx <- as.integer(x)
    
    if( newx != x ) 
    {
        message <- sprintf("Type of %s is not integer\n", x)
        stop(message)
    }
}
