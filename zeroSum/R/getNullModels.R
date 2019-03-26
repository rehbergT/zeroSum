#' Description of getNullModel function
#'
#' This function calculates a null model for given y,w
#'
#' @return different terms and finale cost of the objective function
#'
#' @keywords internal
#'
getLogisticNullModel <- function(y, worg) {
    out <- list()
    out$beta0 <- -log(sum(worg) / as.numeric(worg %*% y) - 1)
    out$p <- 1.0 / (1.0 + exp(-out$beta0))
    out$w <- out$p * (1 - out$p)
    out$z <- out$beta0 + (y - out$p) / out$w
    out$w <- out$w * worg

    return(out)
}

getMultinomialNullModel <- function(y, worg, iterations) {
    out <- list()
    N   <- nrow(y)
    K   <- ncol(y)

    out$beta0 <- rep(0, K)
    for (ii in 1 : iterations) {
        for (k in 1: K) {
            out$beta0[k] <- -log((sum(worg) / as.numeric(worg %*% y[, k]) - 1) /
                            (sum(exp(out$beta0)[-k])))
        }
    }
    out$beta0 <- as.numeric(scale(out$beta0, center = TRUE, scale = FALSE))

    out$p <- exp(out$beta0) / sum(exp(out$beta0))
    w <- out$p * (1 - out$p)

    out$z <- out$beta0 + (y - out$p) / w

    out$z <- matrix(0.0, nrow = N, ncol = K)
    out$w <- matrix(0.0, nrow = N, ncol = K)
    for (k in 1: K) {
        out$z[, k] <- out$beta0[k] + (y[, k] - out$p[k]) / w[k]
        out$w[, k] <- worg * w[k]
    }

    return(out)
}

getCoxNullModel <- function(y, status, worg) {
    N  <- nrow(y)

    ## find ties with events
    y <- cbind(y, status)
    y <- cbind(y, duplicated(y[, 1] + y[, 2]))

    d <- rep(0, N)
    ## calculate d for each set D_i
    j <- 1
    while (j <= N) {
        ## skip if there is no event
        if (y[j, 2] == 0) {
            j <- j + 1
            next
        }

        d[j] <- worg[j]
        k <- j + 1
        ## search for duplicates (ties) and add the weights
        while (k <= N && y[k, 3] == 1) {
            if (y[k, 2] == 1)
                d[j] <- d[j] + worg[k]
            k <- k + 1
        }
        j <- k
    }


    ## calculate u for each sample (omit duplicates and non events)
    u <- rep(0.0, N)
    u[1] <- 1.0
    for (i in 2:N) u[i] <- u[i - 1] - worg[i - 1]

    wr <- rep(0, N)
    du <- d / u
    du[d == 0.0] <- 0.0

    for (i in 1:N) {
        k <- 1
        tmp <- 0
        while (k <= i) {
            if (d[k] != 0.0) tmp <- tmp + worg[i] * du[k]
            k <- k + 1
        }
        wr[i] <- worg[i] * y[i, 2] - tmp
    }

    return(as.matrix(wr))
}
