#' Description of vectorElNetCostFunction function
#'
#' This function calculates the cost function for the elastic net regression.
#'
#' @param x data as numeric matrix object (rows=samples)
#'
#' @param y response vector (length(y)==nrow(x))
#'
#' @param beta coefficients of the linear model
#'
#' @param lambda penalizing parameter of the elastic-net regularization
#'
#' @param alpha Lasso/Ridge adjustment: For alpha = 0 the elastic net becomes
#'              a ridge regularization, for alpha = 1 the elastic net becomes
#'              the lasso regularization
#'
#' @return different terms and finale cost of the objective function
#'
#' @keywords internal
#'
#' @export
vectorElNetCostFunction <- function( x, y, beta , lambda, alpha)
{        
    if( is.null(colnames(x)) || colnames(x)[1] != "Intercept" )
    {
        xin <- cbind( rep( 1.0, nrow(x)), x)
        
    } else
    {
        xin <- x
    }
    
    out <- list()
    # calculation of the residuals
    res <-  y - xin %*% beta 
    
    # calculation of the residual sum of squares 
    out$rss <- sum( (res)^2 )    
    
    # calculation of the ridge penalty (the offset is not penalized
    # therefore beta[1] is excluded)
    out$ridge <- sum( (beta[-1])^2 )
    
    # calculation of the lasso penalty (the offset is not penalized
    # therefore beta[1] is excluded)
    out$lasso <- sum( abs(beta[-1])) 
    
    # calculation of the cost function
    N <- nrow(xin)
    out$cost <- out$rss / (2*N) 
    out$cost <- out$cost + lambda * ((1-alpha)*out$ridge/2 + alpha * out$lasso)
    
    return(out)
}
    
