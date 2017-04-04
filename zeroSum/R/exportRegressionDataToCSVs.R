#' Description of exportRegressionDataToCSV function
#' exports all zeroSum settings for using the c version of zeroSum
#'
#' @keywords internal
#'
#' @export
exportRegressionDataToCSV <- function(
            x,
            y,
            lambda              = 0,
            lambdaSteps         = 100,
            alpha               = 1.0,
            weights             = NULL,
            penalty.factor      = NULL,
            zeroSumWeights      = NULL,
            cSum                = 0.0,
            standardize         = TRUE,
            gamma               = 0.0,
            gammaSteps          = 1,
            fusion              = NULL,
            epsilon             = 0.001,
            nFold               = 10,
            foldid              = NULL,
            useOffset           = TRUE,
            useApprox           = TRUE,
            downScaler          = 1,
            verbose             = FALSE,
            type                = "gaussianZS",
            algorithm           = "CD",
            precision           = 1e-8,
            diagonalMoves       = TRUE,
            lambdaScaler        = 1,
            polish              = 0,
            cvStop              = 0.1,
            path                = "",
            name                = "" )
{
    data <- regressionObject(x, y, NULL, lambda, alpha,
                gamma, cSum, type, weights, zeroSumWeights,
                penalty.factor, fusion, precision,
                useOffset, useApprox, downScaler, algorithm,
                diagonalMoves, polish, standardize, lambdaSteps,
                gammaSteps, nFold, foldid, epsilon, cvStop, verbose,
                lambdaScaler)

    utils::write.csv( data$x, file=paste0( path, "x", name, ".csv" ) )
    utils::write.csv( data$y, file=paste0( path, "y", name, ".csv" ) )

    if( !is.null(data$fusionC) && data$type %in% zeroSumTypes[c(3,4,7,8,11,12),2] )
        utils::write.csv( data$fusionC, file=paste0( path, "fusion", name, ".csv" ) )

    cols <- max(  length(data$lambda), length(data$gamma), ncol(x), nrow(x) )
    settings <- matrix( NA, nrow=22, ncol=cols)
    rownames(settings) = c( 'type', 'N', 'P', 'K', 'nc', 'cSum', 'alpha',
            'diagonalMoves', 'useOffset', 'useApprox', 'precision', 'algorithm',
            'verbose', 'w', 'u', 'v', 'lambda', 'gamma', 'nFold', 'foldid','downScaler', 'cvStop')

    settings[1,1]  <- data$type
    settings[2,1]  <- nrow(data$x)
    settings[3,1]  <- ncol(data$x)
    settings[4,1]  <- ncol(data$y)
    settings[5,1]  <- data$nc
    settings[6,1]  <- data$cSum
    settings[7,1]  <- data$alpha
    settings[8,1]  <- data$diagonalMoves
    settings[9,1]  <- data$useOffset
    settings[10,1] <- data$useApprox
    settings[11,1] <- data$precision
    settings[12,1] <- data$algorithm
    settings[13,1] <- data$verbose

    settings[14,1:nrow(data$x)] <- data$w
    settings[15,1:ncol(data$x)] <- data$u
    settings[16,1:ncol(data$x)] <- data$v

    settings[17,1:length(data$lambda)] <- data$lambda
    settings[18,1:length(data$gamma)]  <- sample(data$gamma)

    settings[19,1] <- data$nFold
    settings[20,1:nrow(data$x)] <- data$foldid
    settings[21,1] <- data$downScaler

    settings[22,1] <- data$cvStop;
    utils::write.csv( settings, file=paste0( path, "settings", name, ".csv" ), na='' )
    saveRDS( data, file=paste0( path, "rDataObject_", name, ".rds"))

}
