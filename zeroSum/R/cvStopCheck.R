#' Description of devianceStop function
#'
#' This function checks if the CV-Log-Likelihood got worse in the last steps
#'
#' @return true or false
#'
#' @keywords internal
#'
cvStopCheck <- function( cvError, maxWorse )
{
    l <- length(cvError)    
    if(l <= maxWorse || l < 2) return(FALSE)

    lastCVError <- cvError[ (l-maxWorse+1):l ]    
    diffs <-  lastCVError[ -1 ] - lastCVError[ -length(lastCVError) ]

    if( sum( diffs >= 0 ) == 0 )
    {
        return(TRUE)       
    } else   
    {
        return(FALSE)
    }
}



