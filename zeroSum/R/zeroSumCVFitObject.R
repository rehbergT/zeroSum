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
                                alpha, 
                                type, 
                                algorithmCV, 
                                algorithmAllSamples, 
                                coefs,
                                lambdaMin,
                                lambda1SE,
                                offset, 
                                precision, 
                                diagonalMoves,
                                polish, 
                                numberOfBetas,
                                logLikelihoodCV,
                                logLikelihood,
                                logLikelihoodCVSD
        )
{
    zeroSumCVFit <- list()

    zeroSumCVFit$lambdaSeq <- lambdaSeq  

    zeroSumCVFit$LambdaMinIndex <- lambdaMin    
    zeroSumCVFit$Lambda1SEIndex <- lambda1SE
    zeroSumCVFit$LambdaMin <- lambdaSeq[lambdaMin]
    zeroSumCVFit$Lambda1SE <- lambdaSeq[lambda1SE]
    
    zeroSumCVFit$coefs <- coefs
    zeroSumCVFit$numberOfBetas <- numberOfBetas
    
    zeroSumCVFit$algorithmCV <- algorithmCV
    zeroSumCVFit$ algorithmAllSamples <-  algorithmAllSamples

    zeroSumCVFit$alpha <- alpha

    zeroSumCVFit$offset <- offset
    zeroSumCVFit$diagonalMoves <- diagonalMoves
    zeroSumCVFit$polish <- polish
    zeroSumCVFit$type <- type
    zeroSumCVFit$logLikelihoodCV <- logLikelihoodCV
    zeroSumCVFit$logLikelihoodCVSD <- logLikelihoodCVSD
    zeroSumCVFit$logLikelihood <- logLikelihood
    
    class(zeroSumCVFit) <- append( class(zeroSumCVFit), "ZeroSumCVFit")
    return(zeroSumCVFit) 

}
