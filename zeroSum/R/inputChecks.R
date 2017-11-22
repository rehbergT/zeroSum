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
                        "fusionGaussian",
                        "fusionGaussianZS",
                        "binomial",
                        "binomialZS",
                        "fusionBinomial",
                        "fusionBinomialZS",
                        "multinomial",
                        "multinomialZS",
                        "fusionMultinomial",
                        "fusionMultinomialZS",
                        "cox",
                        "coxZS",
                        "fusionCox",
                        "fusionCoxZS"
                     ),
                    1:16,
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

checkSurvialDataVector <- function( x, varName)
{
    if( NCOL(x) != 2 )
    {
        message <- sprintf("%s does not consist of two columns", varName)
        stop(message)
    }
    checkNumericVector(x[,1], varName)
    checkBinominalVector(x[,2], varName)
}

checkData <- function( x, y, w, type)
{
    x <- checkNumericMatrix(x, 'x')
    if(is.null(colnames(x)))
        colnames(x) <- as.character(seq(1, ncol(x)))

    N <- nrow(x)

    if( is.null(w)) {
        w <- rep( 1/N, N)
    } else {
        checkNonNegativeNonZeroWeights(w, N, "weights")
    }
    w <- w / sum(w)

    if( nrow(x) != nrow(as.matrix(y)) )
        stop("nrow(x) != nrow(y) !")

    if( type %in% zeroSumTypes[1:4,1] )
    {
        checkNumericVector(y, "y")
        return( list( x=x,
                      y=as.matrix(y),
                      w=w ) )

    } else if( type %in% zeroSumTypes[5:8,1] )
    {
        checkBinominalVector(y, "y")
        return( list( x=x,
                      y=as.matrix(as.numeric(y)),
                      w=w ) )

    } else if( type %in% zeroSumTypes[9:12,1] )
    {
        checkMultinominalVector(y, "y")
        N <- length(y)
        K <- max(y)
        ymatrix <- matrix(0.0, nrow=N, ncol=K )
        for( i in 1:N )
        {
            ymatrix[ i, y[i] ] <- 1.0
        }
        return( list( x=x,
                      y=ymatrix,
                      w=w ) )

    } else if( type %in% zeroSumTypes[13:16,1] )
    {
        checkSurvialDataVector(y, "y")
        ord <- order(y[,1], y[,2] )
        y <- as.matrix(y)
        x <- x[ord, ]
        y <- y[ord, ]
        w <- w[ord]

        return( list( x=x,
                      y=as.matrix(y[,1]), status=as.integer(y[,2,drop=FALSE]),
                      w=w ) )
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
    checkNumericVector( x, name)
    if( length(x) != n )
    {
        message <- sprintf("Length of %s is not correct!", name)
        stop(message)
    }
}

checkNonNegativeWeights <- function( x, n, name)
{
    checkNumericVector( x, name)
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

checkNonNegativeNonZeroWeights <- function( x, n, name)
{
    checkNumericVector( x, name)
    if( length(x) != n )
    {
        message <- sprintf("Length of %s is not correct!", name)
        stop(message)
    }
    if( any(x<=0) )
    {
        message <- sprintf("%s are not allowed to be negative!", name)
        stop(message)
    }
}
