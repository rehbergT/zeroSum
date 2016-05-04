
    context("Testing lambda max calculation")

    test_that( "lambda max calculation works",  {

        lambdaMax <- function( x, beta0, alpha)
        {
            maxRes <- -.Machine$double.xmax
            ind <- 1
            P <- ncol(x)

            for(k in 1:P)
            {
                for(s in 1:k)
                {
                    if(s==k) next
                    a <- abs(sum( ( x[,s] - x[,k] ) * beta0 ))                   
                    if( a > maxRes ) maxRes <- a
                }
            }
            lambdaMax <- maxRes / ( 2.0 * N * alpha ) 

            return(lambdaMax) 
        }

        N <- 27
        P <- 113
        x <- matrix( rnorm(N*P), nrow=N, ncol=P )
        beta0 <- rnorm(N)
        alpha <- 0.71

        expect_that( .Call("LambdaMax", x, beta0, alpha),
                     equals( lambdaMax( x, beta0, alpha) )  )
        
        

    })
