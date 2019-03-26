#' Description of regresion object
#'
#' @import Matrix
#'
#' @useDynLib zeroSum, .registration = TRUE
#'
#' @keywords internal
regressionObject <- function(x, y, beta, alpha, lambda, gamma=0.0,
        type=zeroSumTypes[1, 1], weights=NULL, penalty.factor=NULL,
        zeroSumWeights=NULL, fusion=NULL, precision=1e-8, useIntercept=TRUE,
        useApprox=TRUE, downScaler=1, algorithm="CD", rotatedUpdates=TRUE,
        usePolish=TRUE, standardize=TRUE, lambdaSteps=1, gammaSteps=1, nFold=1,
        foldid=NULL, epsilon=NULL, cvStop = 0.1, verbose=FALSE, threads=1,
        center=TRUE, useZeroSum) {
    data <- list()
    checkType(type)
    id <- which(zeroSumTypes[, 1] == type)
    data$type <- as.integer(zeroSumTypes[id, 2])
    data$useZeroSum <- as.integer(useZeroSum)

    tmp <- checkData(x, y, weights, type)
    data$x <- tmp$x
    data$y <- tmp$y
    data$w <- tmp$w

    N <- nrow(data$x)
    P <- ncol(data$x)

    data$K <- NCOL(data$y)
    data$P <- P
    data$N <- N

    data$cSum  <- 0.0
    data$downScaler <- checkDouble(downScaler,  "downScaler")
    data$alpha <- checkDouble(alpha, "alpha")

    if (threads == "auto") {
        data$threads <- parallel::detectCores()
    } else {
        data$threads <- checkInteger(threads, "threads")
    }

    if (data$type == zeroSumTypes[4, 2]) {
        data$K <- 1
        useIntercept <- as.integer(FALSE)
        data$status <- tmp$status
    }

    if (is.null(penalty.factor)) {
        penalty.factor <- rep(1, P)
    } else {
        checkNonNegativeWeights(penalty.factor, P, "penalty.factors")
    }
    data$v <- penalty.factor

    if (is.null(zeroSumWeights)) {
        zeroSumWeights <- rep(1, P)
    } else {
        checkNonNegativeWeights(zeroSumWeights, P, "zeroSumWeights")
    }
    data$u <- zeroSumWeights
    data$gammaSteps  <- checkInteger(gammaSteps, "gammaSteps")
    data$gamma       <- checkDouble(gamma, "gamma")

    if (length(gamma) == 1 && gamma == 0.0) {
        data$useFusion <- as.integer(FALSE)
    } else {
        data$useFusion <- as.integer(TRUE)
    }

    ## if a lambda sequence is supplied then use the supplied lambdas
    ## if only one lambda is supplied no cv needed to determine lambda
    data$lambdaSteps <- checkInteger(lambdaSteps, "lambdaSteps")
    if (!is.null(lambda)) {
        data$lambda <- checkDouble(lambda, "lambda")
        data$lambdaSteps <- length(data$lambda)
    } else {
        ## set to nan mean it is approximated below
        data$lambda <- NaN
    }

    ## if the number of folds is selected by the user use this value
    ## if only one lambda value is given no cv is necessary -> nfold set to 0
    ## else use 10 folds as default
    if (!is.null(nFold)) {
        data$nFold <- checkInteger(nFold, "nFold")
    } else if (length(data$lambda) == 1 && !any(is.nan(data$lambda))) {
        data$nFold <- 0
    } else {
        data$nFold <- 10
    }

    ## check if more cv folds than samples are selected
    if (data$nFold > N) {
        print(paste0("more CV folds (default nFold=10) than sample size! ",
                     "Setting nFold to ", N))
        data$nFold <- N
    }

    if (is.null(foldid)) {
        if (data$type == zeroSumTypes[4, 2]) {
            nonCensored <- sum(data$status == 1)

            if (nonCensored < data$nFold) {
                print(paste0("more CV folds (default nFold=10) than ",
                            "non-censored ! Setting nFold to ", nonCensored))
                data$nFold <- nonCensored
            }

            data$foldid <- rep(0, N)
            data$foldid[data$status == 1] <- sample(rep(rep(1:data$nFold),
                                            length.out = nonCensored))
            data$foldid[data$status == 0] <- sample(rep(rep(1:data$nFold),
                                            length.out = data$N - nonCensored))
        }else{
            data$foldid <- sample(rep(1:data$nFold, length.out = N))
        }
    }else{
        if (!is.null(tmp$ord))
            foldid <- foldid[tmp$ord]

        if (length(foldid) != N)
            stop("invalid fold numbering (( length(foldid) != N ))\n")

        data$nFold <- max(foldid)
        data$foldid <- foldid
    }

    data$foldid <- as.integer(data$foldid)
    data$nFold  <- as.integer(data$nFold)

    if (data$nFold != 0)
        checkFolds(data$status, data$foldid, data$nFold, data$type)

    if (is.null(fusion)) {
        data$fusion <- NULL
        data$nc     <- as.integer(0)
    }else{
        checkSparseMatrix(fusion, "fusion")
        data$fusion <- fusion
        data$nc     <- as.integer(nrow(fusion))
    }

    data$useIntercept <- checkInteger(useIntercept)
    data$center <- as.integer(center)

    if (standardize == TRUE) {
        data$standardize <- as.integer(TRUE)
    } else{
        data$standardize <- as.integer(FALSE)
    }

    data$verbose <- as.integer(verbose)

    if (any(is.nan(data$lambda))) {
        # in the ridge case (alpha==0) lambdaMax can not be determined
        # therefore a small alpha is used to determine a lambda max
        # the variable ridge is used as a bool to revert alpha to zero
        # after the calculation
        if (verbose) print("Determine lambdaMax")

        ridge <- FALSE

        if (data$alpha == 0.0) {
            data$alpha <- 0.01
            ridge <- TRUE
        }
        lambdaMax <- NULL
        nM  <- NULL
        res <- NULL

        if (data$type == zeroSumTypes[1, 2]) {
            nM  <- sum(data$y * data$w)
            res <- as.matrix(data$y - nM, ncol = 1) * data$w

        }else if (data$type == zeroSumTypes[2, 2]) {
            nM  <- getLogisticNullModel(data$y, data$w)
            res <- as.matrix((nM$z - nM$beta0) * nM$w, ncol = 1)

        } else if (data$type == zeroSumTypes[3, 2]) {
            nM <- getMultinomialNullModel(data$y, data$w, 10)
            res <- matrix(0, ncol = ncol(nM$z), nrow = nrow(nM$z))
            for (i in 1:nrow(res))
                res[i, ] <- (nM$z[i, ] - nM$beta0) * nM$w[i, ]

        }  else if (data$type == zeroSumTypes[4, 2]) {
            res <- getCoxNullModel(data$y, data$status, data$w)
        }

        vtmp <- data$v

        if (data$standardize) {
            sw <- 1.0 / sum(data$w)
            wm <- (data$w %*% data$x) * sw
            for (i in 1:P)
                vtmp[i] <- data$v[i] * sqrt(((data$x[, i] - wm[i])^2 %*%
                                                                data$w) * sw)
        }

        if (data$useZeroSum) {
            lambdaMax <- .Call("lambdaMax", data$x, res, data$u, vtmp,
                                data$alpha, PACKAGE = "zeroSum")
        }else{
            lambdaMax <- max((abs(t(data$x) %*% res)
                                        / (vtmp * data$alpha))[vtmp != 0])
        }

        if (!is.null(epsilon)) {
            epsilon <- checkDouble(epsilon)
        } else {
            if (N < P) {
                epsilon <- 0.01
            } else {
                epsilon <- 0.0001
            }

        }

        # lambdaMin is calculated with the epsilon paramter
        lambdaMin <- epsilon * lambdaMax

        # the lambda sequence is constructed by equally distributing lambdaSteps
        # value on the lineare log scale between lambdaMin and lambdaMax
        data$lambda <- exp(seq(log(lambdaMax), log(lambdaMin),
                            length.out = data$lambdaSteps))

        # revert alpha to zero in the ridge case
        if (ridge == TRUE) {
            data$alpha <- 0.0
        }
    }

    if (length(data$gamma) == 1 && data$gammaSteps > 1) {
        print("approx gamma todo")

    }else if (length(data$gamma) > 1) {
        data$gammaSteps <- length(data$gamma)
    }


    if (cvStop == FALSE)
        cvStop <- 1.0;

    data$cvStop <- checkDouble(cvStop, "CV-Stop")
    if (data$cvStop < 0 || data$cvStop > 1)
        stop("cvStop is not within [0,1]\n")
    data$cvStop <- as.integer(round(data$lambdaSteps * data$cvStop))

    if (is.null(data$fusion)) {
        data$fusionC <- NULL
    }else{
        data$fusionC <- as.matrix(Matrix::summary(data$fusion))
        data$fusionC <- data$fusionC[order(data$fusionC[, 2]), ]
        data$fusionC[, 1:2] <- data$fusionC[, 1:2] - 1 ## C starts with 0
    }

    if (is.null(beta)) {
        cols <- data$K
        if (data$type == zeroSumTypes[4, 2]) cols <- 1
        cols <- cols * (data$nFold + 1)

        beta <- matrix(0, ncol = cols, nrow = data$P + 1)
    }else{
        beta <- as.matrix(beta)
        if (nrow(beta) != ncol(x) + 1) {
            stop("Length of betas doesn't match ncol(x)!")
        }
        if (ncol(beta) != (data$nFold + 1) * data$K) {
            stop("Ncol of beta must be equal to the number of CV folds +1 ",
                 "(for creating the final model using all samples)!")
        }
        if (data$useZeroSum) {
            if (!any(as.numeric(beta[-1, ] %*% data$u) != data$cSum)) {
                stop("Sum of betas doesn't match cSum!")
            }
        }

    }

    data$beta <- beta
    rownames(data$beta) <- c("Intercept", colnames(data$x))

    data$precision <- checkDouble(precision, "precision")

    checkAlgo(algorithm)
    id <- which(zeroSumAlgos[, 1] == algorithm)
    data$algorithm <- as.integer(zeroSumAlgos[id, 2])


    if (data$useFusion) {
        data$algorithm   <- as.integer(3)
    }

    data$useApprox <- checkInteger(useApprox)

    if (data$type == zeroSumTypes[1, 2]) {
        data$useApprox <- as.integer(FALSE)
    }else if (data$algorithm == 1) {
        data$useApprox <- as.integer(TRUE)
    }

    data$rotatedUpdates <- checkInteger(rotatedUpdates)
    data$usePolish      <- checkInteger(usePolish)
    data$seed           <- sample(.Machine$integer.max, size = 1)

    return(data)
}
