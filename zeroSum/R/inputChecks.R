#' Description of inputChecks function
#'
#' These function are for checking user input
#'
#' @return true or stop
#'
#' @importFrom methods as
#'
#' @keywords internal
#'
#


zeroSumTypes <- data.frame( c(
                        "gaussian",
                        "gaussianZS",
                        "fusedGaussian",
                        "fusedGaussianZS",
                        "fusionGaussian",
                        "fusionGaussianZS",
                        "binomial",
                        "binomialZS",
                        "fusedBinomial",
                        "fusedBinomialZS",
                        "fusionBinomial",
                        "fusionBinomialZS",
                        "multinomial",
                        "multinomialZS",
                        "fusedMultinomial",
                        "fusedMultinomialZS",
                        "fusionMultinomial",
                        "fusionMultinomialZS" ),
                    1:18,
                    stringsAsFactors = FALSE )

colnames(zeroSumTypes) <- c("Type", "Int")


zeroSumAlgos <- data.frame( c(
                        "CD",
                        "SA",
                        "LS",
                        "CD+LS" ),
                    c(1,2,3,4),
                    stringsAsFactors = FALSE )

colnames(zeroSumAlgos) <- c("Algo", "Int")


checkNumericMatrix <- function( x, varName)
{
    if( class(x) != "matrix" | typeof(x) != "double"  )
    {
        message <- sprintf("Type of %s is not a numeric matrix", varName)
        stop(message)
    }
    return(x)
}

checkSparseMatrix <- function( x, varName)
{
    x <- as(x, "sparseMatrix")
    if( class(x) != "dgCMatrix" | typeof(x) != "S4"  )
    {
        message <- sprintf("Type of %s is not a sparse matrix or cannot be casted to a sparse matrix\n", varName)
        stop(message)
    }    
}

checkNumericVector <- function( x, varName)
{
    if( class(x) != "numeric" | typeof(x) != "double"  ){
        message <- sprintf("Type of %s is not a numeric vector", varName)
        stop(message)
    }
}

checkBinominalVector <- function( x, varName)
{
    x <- as.integer(x)
    if( any( x != 1 & x != 0  ) )
    {
        message <- sprintf("%s does not consist of 0 and 1", varName)
        stop(message)
    }
}

checkMultinominalVector <- function( x, varName)
{
    x <- as.integer(x)
    if( any( !(x %in% (1:max(x)))  ) )
    {
        message <- sprintf("%s does not consist of 1:%d", varName, max(x))
        stop(message)
    }
}


checkResponse <- function( y, varName, type)
{
    if( type %in% zeroSumTypes[1:6,1] )
    {
        checkNumericVector(y, varName)
        return(as.matrix(y))

    } else if( type %in% zeroSumTypes[7:12,1] )
    {
        checkBinominalVector(y, varName)
        return(as.matrix(as.numeric(y)))

    } else if( type %in% zeroSumTypes[13:18,1] )
    {
        checkMultinominalVector(y, varName)
        N <- length(y)
        K <- max(y)
        ymatrix <- matrix(0.0, nrow=N, ncol=K )
        for( i in 1:N )
        {
            ymatrix[ i, y[i] ] <- 1.0
        }
        return(ymatrix)
    }
}

checkType <- function( type )
{
    if( class(type) != "character" & typeof(type) != "character" |
        !( type %in% zeroSumTypes[,1] ) )
    {
        message <- paste0( "Selected type is not validUse zeroSumElNet ",
                              "(default), zeroSumLogistic or elNet!")
        stop(message)
    }
}

checkAlgo <- function( algo, name)
{
    if( class(algo) != "character" &  typeof(algo) != "character"
        | !(algo %in% zeroSumAlgos[,1]) )
    {
        message <- sprintf( "Selected %s is not valid")
        stop(message)
    }
}

checkDouble <- function( x, name)
{
    if( class(x) != "numeric" | typeof(x) != "double"   )
    {
        message <- sprintf("Type of %s is not numeric", name)
        stop(message)
    }
    return(as.numeric(as.numeric(x)))
}

checkInteger <- function( x, name)
{
    newx <- as.integer(x)

    if( newx != x )
    {
        message <- sprintf("Type of %s is not integer", name)
        stop(message)
    }
    return(newx)
}

checkWeights <- function( x, n, name)
{
    checkNumericVector <- function( x, name)
    if( length(x) != n )
    {
        message <- sprintf("Length of %s is not correct!", name)
        stop(message)
    }
}

checkNonNegativeWeights <- function( x, n, name)
{
    checkNumericVector <- function( x, name)
    if( length(x) != n )
    {
        message <- sprintf("Length of %s is not correct!", name)
        stop(message)
    }
    if( any(x<0) )
    {
        message <- sprintf("%s are not allowed to be negative!", name)
        stop(message)
    }
}
