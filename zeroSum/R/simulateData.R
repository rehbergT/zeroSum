#' Description of the simulateData function
#'
#' This function simulates a data set from a given linear model.
#'
#' @param coefs true coeffients used for simulating a data set
#'
#' @param corBetas correlation Matrix for the supplied beta Matrix
#'
#' @param samples number of samples
#'
#' @param responseSD standard deviation of random noise added 
#'        to the response
#'
#' @param sampleWiseSD standard deviation of the sample-wise shifts 
#'
#' @return simulated data set
#'
#' @examples
#' data <- simulateData()
#' dim( data$x )
#' length( data$y )
#'
#' @export
simulateData <- function(   coefs = c( rnorm(10), rep(0,50)),
                            corBetas = NULL,
                            samples = 20,
                            responseSD = 0.1,
                            sampleWiseSD = 1.0 )
{
    n <- samples
    p <- length(coefs)
    x <- NULL
    
    if( is.null(corBetas) )
    {
        x <- matrix( rnorm(p*n,0,.5), nrow=n, ncol=p)
        colnames(x) <- as.character(paste0("feature_", seq(1, p)))
        rownames(x) <- as.character(paste0("sample_", seq(1, n)))
    } else
    {
        if( ncol(corBetas) != nrow(corBetas) || sum(coefs!=0) > ncol(corBetas) )
        {
            stop("corBetas not valid!")
        }
        
        Ncor <- ncol(corBetas)        
        EV.matrix <- chol(corBetas)   
        random.normal  <- matrix( rnorm( Ncor*n, 0, 0.5), ncol = Ncor, nrow = n )             
        
        x <- matrix( 0, nrow=n, ncol=p)
        
        corValues <- which( coefs!= 0)
        x[,corValues] <- random.normal %*% EV.matrix

        x.remaining <- matrix( rnorm( (p-Ncor)*n, 0, 0.5), ncol = p-Ncor, nrow = n )  
        x[,-corValues] <- x.remaining   
        
    }
        
    y <- as.vector(x %*% coefs)
    y <- y+rnorm(n, mean = 0, sd = responseSD)
    names(y) <- as.character(paste0("sample_", seq(1, n)))


    for(i in 1:nrow(x))
    {
        x[i,] <- x[i,] + rnorm( 1, mean=0, sd=sampleWiseSD )
    }

    out <- list()
    out$x <- x
    out$y <- y
    out$coefs <- coefs

    return( out )
}

