#' Description of inputChecks function
#'
#' These function are for checking user input
#'
#' @return true or stop
#'
#' @import methods
#'
#' @keywords internal
#'
#


zeroSumTypes <- data.frame(c("gaussian", "binomial", "multinomial", "cox"),
    1:4,
    stringsAsFactors = FALSE
)

colnames(zeroSumTypes) <- c("Type", "Int")


zeroSumAlgos <- data.frame(c("CD", "SA", "LS", "CDP"), c(1, 2, 3, 4),
    stringsAsFactors = FALSE
)

colnames(zeroSumAlgos) <- c("Algo", "Int")


checkNumericMatrix <- function(x, varName) {
    if (any(class(x) == "tbl") & typeof(x) == "list") {
        x <- as.matrix(x)
    }

    if (any(class(x) == "data.frame") & typeof(x) == "list") {
        x <- as.matrix(x)
    }

    if (!any(class(x) == "matrix") | typeof(x) != "double") {
        message <- sprintf("%s is not a matrix", varName)
        stop(message)
    }

    if (any(is.na(x))) {
        message <- sprintf("%s contains NA values!", varName)
        stop(message)
    }

    if (any(is.nan(x))) {
        message <- sprintf("%s contains NaN values!", varName)
        stop(message)
    }

    if (any(is.infinite(x))) {
        message <- sprintf("%s contains NaN values!", varName)
        stop(message)
    }

    return(x)
}

checkSparseMatrix <- function(x, varName) {
    x <- methods::as(x, "sparseMatrix")
    if (class(x) != "dgCMatrix" | typeof(x) != "S4") {
        message <- sprintf(paste0(
            "Type of %s is not a sparse matrix or cannot",
            "be casted to a sparse matrix\n"
        ), varName)
        stop(message)
    }
}

checkNumericVector <- function(x, varName) {
    if (any(class(x) == "tbl") & typeof(x) == "list") {
        x <- as.matrix(x)
    }

    if (any(class(x) == "data.frame") & typeof(x) == "list") {
        x <- as.matrix(x)
    }

    if (any(class(x) == "numeric") & typeof(x) == "double") {
        x <- as.matrix(x)
    }

    if (typeof(x) == "integer") {
        x <- as.matrix(as.numeric(x))
    }

    if (!any(class(x) == "matrix") | typeof(x) != "double" | ncol(x) > 1) {
        message <- sprintf("%s is not a vector", varName)
        stop(message)
    }

    if (any(is.na(x))) {
        message <- sprintf("%s contains NA values!", varName)
        stop(message)
    }

    if (any(is.nan(x))) {
        message <- sprintf("%s contains NaN values!", varName)
        stop(message)
    }

    if (any(is.infinite(x))) {
        message <- sprintf("%s contains NaN values!", varName)
        stop(message)
    }

    return(x)
}

checkBinominalVector <- function(x, varName) {
    x <- as.integer(x)
    if (any(x != 1 & x != 0)) {
        message <- sprintf("%s does not consist of 0 and 1", varName)
        stop(message)
    }
}

checkMultinominalVector <- function(x, varName) {
    x <- as.integer(x)
    if (any(!(x %in% (1:max(x))))) {
        message <- sprintf("%s does not consist of 1:%d", varName, max(x))
        stop(message)
    }
}

checkSurvialDataVector <- function(x, varName) {
    if (NCOL(x) != 2) {
        message <- sprintf("%s does not consist of two columns", varName)
        stop(message)
    }
    checkNumericVector(x[, 1], varName)
    checkBinominalVector(x[, 2], varName)
}

checkData <- function(x, y, w, type) {
    x <- checkNumericMatrix(x, "x")
    if (is.null(colnames(x))) {
        colnames(x) <- as.character(seq(1, ncol(x)))
    }

    N <- nrow(x)
    if (is.null(w)) {
        w <- rep(1 / N, N)
    } else {
        checkNonNegativeNonZeroWeights(w, N, "weights")
        w <- w / sum(w)
    }

    status <- NULL
    ord <- NULL

    if (type == zeroSumTypes[1, 1]) {
        y <- checkNumericVector(y, "y")
    } else if (type == zeroSumTypes[2, 1]) {
        checkBinominalVector(y, "y")
        y <- as.matrix(as.numeric(y))
    } else if (type == zeroSumTypes[3, 1]) {
        checkMultinominalVector(y, "y")
        N <- length(y)
        K <- max(y)
        ymatrix <- matrix(0.0, nrow = N, ncol = K)
        for (i in 1:N) {
            ymatrix[i, y[i]] <- 1.0
        }
        y <- ymatrix
    } else if (type == zeroSumTypes[4, 1]) {
        checkSurvialDataVector(y, "y")
        y <- as.matrix(y)

        ## sort
        ord <- order(y[, 1], y[, 2])

        ## remove censored samples with time lower than the first event
        i <- 1
        while (y[ord[i], 2] == 0) i <- i + 1

        ord <- ord[i:length(ord)]
        status <- as.integer(y[ord, 2, drop = FALSE])
        y <- y[ord, 1, drop = FALSE]
        x <- x[ord, ]
        w <- w[ord]
    }

    if (nrow(x) != NROW(y)) {
        stop("nrow(x) != nrow(y) !")
    }

    return(list(x = x, y = y, w = w, status = status, ord = ord))
}

checkType <- function(type) {
    if (class(type) != "character" & typeof(type) != "character" |
        !(type %in% zeroSumTypes[, 1])) {
        message <- paste0(
            "Selected type is not valid. Use gaussian, binomial",
            ", multinomial or cox!"
        )
        stop(message)
    }
}

checkAlgo <- function(algo, name) {
    if (class(algo) != "character" & typeof(algo) != "character"
    | !(algo %in% zeroSumAlgos[, 1])) {
        message <- sprintf("Selected %s is not valid")
        stop(message)
    }
}

checkDouble <- function(x, name) {
    if (class(x) != "numeric" | typeof(x) != "double") {
        message <- sprintf("Type of %s is not numeric", name)
        stop(message)
    }
    return(as.numeric(as.numeric(x)))
}

checkInteger <- function(x, name) {
    newx <- as.integer(x)

    if (newx != x) {
        message <- sprintf("Type of %s is not integer", name)
        stop(message)
    }
    return(newx)
}

checkWeights <- function(x, n, name) {
    checkNumericVector(x, name)
    if (length(x) != n) {
        message <- sprintf("Length of %s is not correct!", name)
        stop(message)
    }
}

checkNonNegativeWeights <- function(x, n, name) {
    checkNumericVector(x, name)
    if (length(x) != n) {
        message <- sprintf("Length of %s is not correct!", name)
        stop(message)
    }
    if (any(x < 0)) {
        message <- sprintf("%s are not allowed to be negative!", name)
        stop(message)
    }
}

checkNonNegativeNonZeroWeights <- function(x, n, name) {
    checkNumericVector(x, name)
    if (length(x) != n) {
        message <- sprintf("Length of %s is not correct!", name)
        stop(message)
    }
    if (any(x <= 0)) {
        message <- sprintf("%s are not allowed to be negative!", name)
        stop(message)
    }
}

checkFolds <- function(status, foldid, nFold, type) {
    check <- TRUE
    if (any(!(1:nFold %in% foldid))) check <- FALSE

    if (type %in% zeroSumTypes[13:16, 2]) {
        for (i in 1:nFold) {
            if (!any(status[foldid == i] != 0)) check <- FALSE
        }
    }

    if (check == FALSE) {
        message <- sprintf("Wrong specified foldids!")
        stop(message)
    }
}