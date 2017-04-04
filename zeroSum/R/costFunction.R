#' Description of costFunction function
#'
#' @keywords internal
#'
#' @export
costFunction <- function( data )
{

    out <- list()
    x   <- data$x
    N   <- nrow(x)
    y   <- data$y
    beta <- data$beta
    weights <- data$w
    penalty.factor <- data$v

    if( data$type %in% zeroSumTypes[1:6,2] )
    {
        ## calculation of the residuals
        xTimesBeta <- x %*% beta[-1] + beta[1]

        ## calculation of the residual sum of squares
        res <-  y - xTimesBeta
        out$loglikelihood <- -as.numeric( weights %*% (res^2) ) / 2

    } else if( data$type %in% zeroSumTypes[7:12,2] )
    {
        ## calculation of the residuals
        xTimesBeta <- x %*% beta[-1] + beta[1]

        ## calculation of the loglikelihood
        expXB <- log( 1 + exp( xTimesBeta ) )
        out$loglikelihood <- as.numeric( weights %*% (y * xTimesBeta - expXB ))

    } else if( data$type %in% zeroSumTypes[13:18,2] )
    {
        xb <- x %*% beta[-1,]
        for( i in 1:ncol(beta))
        {
           xb[,i] <- xb[,i] + rep(beta[1,i],N)
        }
        out$loglikelihood <- as.numeric( weights %*% ( rowSums(xb * y) - log(rowSums(exp(xb))) ))
    }

    beta <- as.matrix(beta)
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
    if( data$type %in% zeroSumTypes[c(5,6,11,12),2] )
    {
        out$fusion <- sum(abs(as.numeric( data$fusion %*% data$beta[-1])))
        out$cost <- out$cost + data$gamma * out$fusion

    } else if( data$type %in% zeroSumTypes[c(17,18),2] )
    {
        out$fusion <- 0.0
        for( i in 1:ncol(data$beta))
        {
            out$fusion <- out$fusion + sum(abs(as.numeric( data$fusion %*% data$beta[-1,i])))
        }
        out$cost <- out$cost + data$gamma * out$fusion
    }

    if( data$type %in% zeroSumTypes[c(3,4,9,10),2] )
    {
        beta       <- data$beta[-1] ## remove offset
        fused      <- beta[ -length(beta) ] - beta[-1]
        out$fusion <- sum(abs(fused))
        out$cost   <- out$cost + data$gamma * out$fusion

    } else if( data$type %in% zeroSumTypes[c(15,16),2] )
    {
        out$fusion <- 0.0
        for( i in 1:ncol(data$beta))
        {
            beta       <- as.numeric(data$beta[-1,i]) ## remove offset
            fused      <- beta[ -length(beta) ] - beta[-1]
            out$fusion <- out$fusion + sum(abs(fused))
        }
        out$cost <- out$cost + data$gamma * out$fusion
    }

#     out$test <- .Call( "costFunctionWrapper", data,
#              PACKAGE="zeroSum" )
    return(out)
}
