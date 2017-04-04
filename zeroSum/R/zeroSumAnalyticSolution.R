#' Description of zeroSumAnalyticSolution function
#'
#' Solves the unpenalized least-squares problem for given x and y under the 
#' zero-sum constraint. However this can only be used if p<n, where p=ncol(x)
#' and n=length(y).
#'
#' @param x data as a numeric matrix object (rows=samples). 
#'          The zero-sum regression requires data on the log scale, i.e.
#'          x should be log-transformed data.
#'
#' @param y response vector to be predicted by x (length(y)==nrow(x))
#'
#' @param offset determines if an offset should be used in the model or
#'               not (TRUE/FALSE)
#'
#' @return zeroSumFitObject
#'
#' @examples
#' set.seed(1)
#' x <- log2(exampleData$x+1)
#' y <- exampleData$y
#' fit <- zeroSumAnalyticSolution( x, y )
#'   
#' @export
zeroSumAnalyticSolution <- function( x, y, offset=TRUE )
{
    # some basic checks for the passed arguments
    checkNumericMatrix(x, 'x')
    checkNumericVector(y, 'y')
  
    if( nrow(x) != length(y) )
    {
        stop("number of rows of X does not match length of Y!\n")
    }

    fullP <- NULL
    xNames <- colnames(x)
    if( offset == TRUE && ncol(x) >= nrow(x) )
    {
        cat("Warning: Can not be solved analytical! ")
        cat("More Features than Samples)!\n")
        cat("Better use zeroSumFit! Now using x = [,1:(nrow(x)-1)!\n")

        fullP <- ncol(x)+1
        x <- x[,1:(nrow(x)-1)]  

    } else if ( offset == FALSE && ncol(x) >= nrow(x) )
    {
        cat("Warning: Can not be solved analytical! ")
        cat("More Features than Samples)!\n")
        cat("Better use zeroSumFit! Now using x = [,1:(nrow(x))!\n")
    
        fullP <- ncol(x)
        x <- x[,1:(nrow(x))] 
    }
    
    if( offset ){       
        x <- cbind( rep( 1.0, nrow(x)), x)     
        C <- c(0, rep(1, (ncol(x)-1) ))
    } else{ 
        C <- rep(1, (ncol(x)) )
    }
    
  
    
    CT <- t(C)
    xT <- t(x)
    
    xTx_inv <- solve( xT %*% x )
    
    tmp1 <- xT %*% y 
    
    tmp2 <- solve( CT %*% xTx_inv %*% C )
    tmp3 <- CT %*% xTx_inv %*% xT %*% y
    
    lagrange <- tmp2 * tmp3
    
    betas <- xTx_inv %*% ( tmp1 - lagrange * C )
    betas <- betas[,1]
    
    if( !is.null(fullP) )
    {
        betas <- c( betas, rep( 0, fullP - length(betas) ) )
    }
    
    if( offset ){
        names(betas) <- c("Intercept", xNames )
    }else{
        names(betas) <- xNames
    }
    
    data <- list()
    data$beta <- betas
    class(data) <- append( class(data),"ZeroSumFit")

    return(data)  
}

    
