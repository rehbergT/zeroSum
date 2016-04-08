#' Description of zeroSumCVFitObject function
#'
#' Creates a zeroSumCVFitObject which stores all arguments and results of
#' the zeroSumCVFit() function. 
#'
#' @return zeroSumCVFitObject
#'
#' @keywords internal
#'
zeroSumCVFitObject <- function( lambdaSeq, 
                                CVError, 
                                CVError_SD, 
                                alpha, 
                                devRatio, 
                                type, 
                                algorithmCV, 
                                algorithmAllSamples, 
                                lambdaMin,
                                betaMin, 
                                lambda1SE, 
                                beta1SE, 
                                offset, 
                                precision, 
                                verticalMoves,
                                polish, 
                                numberOfBetas
        )
{
    zeroSumCVFit <- list()

    zeroSumCVFit$lambdaSeq <- lambdaSeq  
    zeroSumCVFit$CVError <- CVError      
    zeroSumCVFit$LambdaMinIndex <- lambdaMin    
    zeroSumCVFit$Lambda1SEIndex <- lambda1SE
    
    zeroSumCVFit$betaMin <- betaMin
    zeroSumCVFit$beta1SE <- beta1SE
    zeroSumCVFit$numberOfBetas <- numberOfBetas
    
    zeroSumCVFit$algorithmCV <- algorithmCV
    zeroSumCVFit$ algorithmAllSamples <-  algorithmAllSamples
    
    zeroSumCVFit$CVError_SD <- CVError_SD  
    zeroSumCVFit$alpha <- alpha
  
    zeroSumCVFit$devRatio <- devRatio
    zeroSumCVFit$offset <- offset
    zeroSumCVFit$verticalMoves <- verticalMoves
    zeroSumCVFit$polish <- polish
    zeroSumCVFit$type <- type
    
    
    class(zeroSumCVFit) <- append( class(zeroSumCVFit), "ZeroSumCVFit")
    return(zeroSumCVFit) 

}
