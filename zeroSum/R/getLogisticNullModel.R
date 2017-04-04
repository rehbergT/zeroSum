#' Description of getLogisticNullModel function
#'
#' This function calculates a null model for given x,y
#'
#' @return different terms and finale cost of the objective function
#'
#' @keywords internal
#'
getLogisticNullModel <- function( y, worg, iterations )
{
    y <- as.numeric(y)
    out <- list()
    N   <- length(y)
    xb  <- 0.0
    for( i in 1 : iterations )
    {
        out$p <- rep( 1 / ( 1 + exp( -xb )), N )
        out$w <- out$p * ( 1 - out$p )
        out$z <- xb + ( y - out$p ) / out$w
        out$w <- out$w * worg
        xb    <- as.numeric( out$z %*% out$w  / sum(out$w))
    }
    out$beta0 <- xb

    return(out)
}

getMultinomialNullModel <- function( y, worg, iterations )
{
    out <- list()
    N   <- nrow(y)
    K   <- ncol(y)
    
    out$beta0 <- rep(0,K)
    out$z     <- matrix(0.0, nrow=N, ncol=K )

    for( ii in 1 : iterations ) 
    {
        tmp <- exp(out$beta0)    
        for( k in 1:K )
            out$p[k] <- tmp[k] / sum(tmp)
        
        tmp <- out$p * ( 1 - out$p )
        out$w <- matrix( tmp, nrow=N, ncol=K, byrow=TRUE)

        for( k in 1:K )
            out$w[,k] <- out$w[,k] * worg
                
        for( i in 1:N )
        {
            for( k in 1:K)
                out$z[i,k] = ( y[i,k] - out$p[k] ) / out$w[i,k] + out$beta0[k]
        }

        for( k in 1:K )
        {
            out$beta0[k] <- as.numeric(( out$w[,k] %*% out$z[,k] ) / sum(out$w[,k]) )
        }
    

    }

    
    return(out)
}
