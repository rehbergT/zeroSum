#' Description of costFunction function
#'
#' @keywords internal
#'
#' @export
costFunction <- function( data, useC=FALSE )
{

    out <- list()
    x   <- data$x
    N   <- nrow(x)
    y   <- data$y
    beta <- data$beta
    weights <- data$w
    penalty.factor <- data$v

    if( data$type %in% zeroSumTypes[1:4,2] )
    {
        ## calculation of the residuals
        xtb <- x %*% beta[-1,] + beta[1,]

        ## calculation of the residual sum of squares
        res <-  y - xtb
        out$loglikelihood <- -as.numeric( weights %*% (res^2) ) / 2

    } else if( data$type %in% zeroSumTypes[5:8,2] )
    {
        xtb <- x %*% beta[-1,] + beta[1,]

        ## calculation of the loglikelihood
        expXB <- log1p(exp(xtb))
        out$loglikelihood <- as.numeric( weights %*% (y * xtb - expXB ))

    } else if( data$type %in% zeroSumTypes[9:12,2] )
    {
        xb <- x %*% beta[-1,]
        for( i in 1:ncol(beta))
        {
           xb[,i] <- xb[,i] + rep(beta[1,i],N)
        }

        xby <- rowSums(xb * y)
        a <- max(xb)
        xb <- xb - a
        out$loglikelihood <- as.numeric( weights %*% ( xby - log(rowSums(exp(xb))) - a ))
    }  else if( data$type %in% zeroSumTypes[13:16,2] )
    {
        y <- cbind( data$y, data$status )
        y <- cbind( y, duplicated(y[,1]+y[,2]) )
        d <- rep(0,N)
        ## calculate d for each set D_i
        j <- 1
        while( j <= N ) {
            ## skip if there is no event
            if( y[j,2] == 0 ) { j <- j+1;  next; }
            d[j] <- weights[j]
            k <- j+1
            ## search for duplicates (ties) and add the weights
            while( k <= N && y[k,3] == 1)
            {
                if( y[k,2] == 1 )
                    d[j] <- d[j] + weights[k]
                k <- k + 1
            }
            j <- k
        }
        id <- d != 0.0
        xtb <- x %*% beta[-1,]

        out$loglikelihood <- sum( (weights * xtb)[y[,2]!=0,] )

        u <- rep(0.0,N)
        for(i in 1:N){
            a <- max(xtb[i:N])
            u[i] <- a + log( sum( weights[i:N] * exp( xtb[i:N] - a ) ) )
        }

        out$loglikelihood <- out$loglikelihood - sum( d[id] * u[id] )
    }

    ## calculation of the ridge penalty (the offset is not penalized
    ## therefore beta[1] is excluded)
    out$ridge <- sum(t( (beta[-1,,drop=FALSE])^2  )  %*% penalty.factor)

    ## calculation of the lasso penalty (the offset is not penalized
    ## therefore beta[1] is excluded)
    out$lasso <- sum(t( abs(beta[-1,,drop=FALSE])  )  %*% penalty.factor)

    ## calculation of the cost function
    out$cost <- -out$loglikelihood + data$lambda *
                    ((1-data$alpha)*out$ridge/2 + data$alpha * out$lasso)

    out$fusion <- 0
    if( data$type %in% zeroSumTypes[c(3,4,7,8,11,12,15,16),2] )
    {
        out$fusion <- 0.0
        for( i in 1:ncol(data$beta))
        {
            out$fusion <- out$fusion + sum(abs(as.numeric( data$fusion %*% data$beta[-1,i])))
        }
        out$cost <- out$cost + data$gamma * out$fusion
    }

    ## this call is for the unit test, which verifies that R and C calculate the same
    if(useC) out$C <- .Call( "costFunctionWrapper", data, PACKAGE="zeroSum" )

    return(out)
}
