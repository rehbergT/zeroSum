

context("Testing zerosum coordinate descent move")

test_that("zerosum coordinate descent move seems to work", {
    data <- regressionObject(
        log2(exampleData$x + 1), exampleData$y, "gaussian",
        0.578, 1, 1, NULL, NULL, NULL, NULL, NULL, NULL, FALSE, TRUE, FALSE,
        1, 0.1
    )

    zeroSumCd <- function(data, k, s) {
        part1 <- data$x[, s] * data$u[k] / data$u[s] - data$x[, k]

        ak <- as.numeric(part1^2 %*% data$w)
        ak <- ak + data$lambda * (1 - data$alpha) * (data$v[k] + data$v[s]
        * data$u[k]^2 / data$u[s]^2)

        part2 <- data$y - data$beta[1] - data$x[, -c(k, s)] %*%
            data$beta[-c(1, k + 1, s + 1)] - data$x[, s] / data$u[s] *
                as.numeric(data$cSum - data$u[-c(s, k)] %*%
                    data$beta[-c(1, s + 1, k + 1)])

        bk <- -sum(part1 * part2 * data$w) + data$lambda *
            (1 - data$alpha) * data$v[s] * (data$u[k] /
                (data$u[s]^2)) * (data$cSum - data$u[-c(s, k)] %*%
                data$beta[-c(1, s + 1, k + 1)])


        bk1 <- (bk - data$lambda * data$alpha * (data$v[k] - data$v[s] *
            data$u[k] / data$u[s])) / ak
        bk2 <- (bk - data$lambda * data$alpha * (data$v[k] + data$v[s] *
            data$u[k] / data$u[s])) / ak
        bk3 <- (bk + data$lambda * data$alpha * (data$v[k] + data$v[s] *
            data$u[k] / data$u[s])) / ak
        bk4 <- (bk + data$lambda * data$alpha * (data$v[k] - data$v[s] *
            data$u[k] / data$u[s])) / ak

        bs1 <- (data$cSum - data$u[-c(s, k)] %*%
            data$beta[-c(1, s + 1, k + 1)] - data$u[k] * bk1) / data$u[s]
        bs2 <- (data$cSum - data$u[-c(s, k)] %*%
            data$beta[-c(1, s + 1, k + 1)] - data$u[k] * bk2) / data$u[s]
        bs3 <- (data$cSum - data$u[-c(s, k)] %*%
            data$beta[-c(1, s + 1, k + 1)] - data$u[k] * bk3) / data$u[s]
        bs4 <- (data$cSum - data$u[-c(s, k)] %*%
            data$beta[-c(1, s + 1, k + 1)] - data$u[k] * bk4) / data$u[s]

        betak <- NA
        betas <- NA
        if (bk1 > 0 && bs1 > 0) {
            betak <- bk1
            betas <- bs1
        } else if (bk2 > 0 && bs2 < 0) {
            betak <- bk2
            betas <- bs2
        } else if (bk3 < 0 && bs3 > 0) {
            betak <- bk3
            betas <- bs3
        } else if (bk4 < 0 && bs4 < 0) {
            betak <- bk4
            betas <- bs4
        }

        return(c(betak, betas))
    }

    k <- 5
    s <- 2

    R <- as.numeric(zeroSumCd(data, k, s))
    C <- .Call("checkMoves", data, as.integer(2), as.integer(k - 1),
        as.integer(s - 1), as.integer(0), as.integer(0),
        PACKAGE = "zeroSum"
    )


    expect_that(R, equals(C, tolerance = 1e-10))
})
