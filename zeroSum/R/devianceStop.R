#' Description of devianceStop function
#'
#' This function checks if the dev ratio got worse in the last steps
#'
#' @return true or false
#'
#' @keywords internal
#'
devianceStop <- function( devRatio, maxWorse )
{
    devRatio <- round( devRatio, digits=3) 
    test <- 0
    i <- length(devRatio)
    while( i>1 && devRatio[i] <=  devRatio[i-1] )
    {
        if( devRatio[i] <=  devRatio[i-1] )        
            test <- test+1
        i <- i-1
    }
    
    if( test >= maxWorse )
    {
        return(TRUE)
    } else
    {
        return(FALSE)
    }
}


