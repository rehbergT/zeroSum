context("Testing rotated zerosum coordinate descent move")

test_that("rotated zerosum coordinate descent move seems to work", {
    data <- regressionObject(
        log2(exampleData$x + 1), exampleData$y, "gaussian",
        0.578, 1, 1, NULL, NULL, NULL, NULL, NULL, NULL, FALSE, TRUE, FALSE,
        1, 0.1
    )

    zeroSumCdRT <- function(data, n, m, s, theta) {
        rtW <- (data$u[n] * cos(theta) - data$u[m] * sin(theta)) / data$u[s]
        part1 <- data$x[, m] * sin(theta) - data$x[, n] * cos(theta) + data$x[, s] * rtW

        a <- as.numeric(part1^2 %*% data$w) + data$lambda * (1 - data$alpha) *
            (data$v[n] * cos(theta)^2 + data$v[m] * sin(theta)^2 + data$v[s] * rtW^2)

        t1 <- data$beta[n + 1]
        t2 <- data$beta[m + 1]

        part2 <- (data$cSum - data$u[-c(n, m, s)] %*% data$beta[-c(1, n + 1, m + 1, s + 1)] -
            data$u[n] * t1 - data$u[m] * t2) / data$u[s]
        part2 <- as.numeric(part2)
        part3 <- data$y - data$beta[1] - data$x[, -c(n, m, s)] %*% data$beta[-c(1, n + 1, m + 1, s + 1)] -
            data$x[, n] * t1 - data$x[, m] * t2 - data$x[, s] * part2


        b <- -sum(part1 * part3 * data$w) - as.numeric(data$lambda * (1 - data$alpha) *
            (data$v[n] * t1 * cos(theta) - data$v[m] * t2 * sin(theta) + data$v[s] * part2 *
                (-data$u[n] * cos(theta) + data$u[m] * sin(theta)) / data$u[s]))

        c1 <- data$v[n] * cos(theta) - data$v[m] * sin(theta) + data$v[s] * (-data$u[n] * cos(theta) + data$u[m] * sin(theta)) * data$v[s] / data$u[s]
        c2 <- data$v[n] * cos(theta) - data$v[m] * sin(theta) + data$v[s] * (data$u[n] * cos(theta) - data$u[m] * sin(theta)) * data$v[s] / data$u[s]
        c3 <- data$v[n] * cos(theta) + data$v[m] * sin(theta) + data$v[s] * (-data$u[n] * cos(theta) + data$u[m] * sin(theta)) * data$v[s] / data$u[s]
        c4 <- data$v[n] * cos(theta) + data$v[m] * sin(theta) + data$v[s] * (data$u[n] * cos(theta) - data$u[m] * sin(theta)) * data$v[s] / data$u[s]
        c5 <- -data$v[n] * cos(theta) - data$v[m] * sin(theta) + data$v[s] * (-data$u[n] * cos(theta) + data$u[m] * sin(theta)) * data$v[s] / data$u[s]
        c6 <- -data$v[n] * cos(theta) - data$v[m] * sin(theta) + data$v[s] * (data$u[n] * cos(theta) - data$u[m] * sin(theta)) * data$v[s] / data$u[s]
        c7 <- -data$v[n] * cos(theta) + data$v[m] * sin(theta) + data$v[s] * (-data$u[n] * cos(theta) + data$u[m] * sin(theta)) * data$v[s] / data$u[s]
        c8 <- -data$v[n] * cos(theta) + data$v[m] * sin(theta) + data$v[s] * (data$u[n] * cos(theta) - data$u[m] * sin(theta)) * data$v[s] / data$u[s]

        betan <- (b - data$lambda * data$alpha * c1) / a * cos(theta) + data$beta[n + 1]
        betam <- (-(b - data$lambda * data$alpha * c1) / a) * sin(theta) + data$beta[m + 1]
        betas <- (data$cSum - data$u[-c(n, m, s)] %*% data$beta[-c(1, n + 1, m + 1, s + 1)] - data$u[n] * betan - data$u[m] * betam) / data$u[s]

        if (betan > 0 && betam > 0 && betas > 0) {
            return(c(betan, betam, betas))
        }
        betan <- (b - data$lambda * data$alpha * c2) / a * cos(theta) + data$beta[n + 1]
        betam <- (-(b - data$lambda * data$alpha * c2) / a) * sin(theta) + data$beta[m + 1]
        betas <- (data$cSum - data$u[-c(n, m, s)] %*% data$beta[-c(1, n + 1, m + 1, s + 1)] - data$u[n] * betan - data$u[m] * betam) / data$u[s]

        if (betan > 0 && betam > 0 && betas < 0) {
            return(c(betan, betam, betas))
        }

        betan <- (b - data$lambda * data$alpha * c3) / a * cos(theta) + data$beta[n + 1]
        betam <- (-(b - data$lambda * data$alpha * c3) / a) * sin(theta) + data$beta[m + 1]
        betas <- (data$cSum - data$u[-c(n, m, s)] %*% data$beta[-c(1, n + 1, m + 1, s + 1)] - data$u[n] * betan - data$u[m] * betam) / data$u[s]

        if (betan > 0 && betam < 0 && betas > 0) {
            return(c(betan, betam, betas))
        }

        betan <- (b - data$lambda * data$alpha * c4) / a * cos(theta) + data$beta[n + 1]
        betam <- (-(b - data$lambda * data$alpha * c4) / a) * sin(theta) + data$beta[m + 1]
        betas <- (data$cSum - data$u[-c(n, m, s)] %*% data$beta[-c(1, n + 1, m + 1, s + 1)] - data$u[n] * betan - data$u[m] * betam) / data$u[s]

        if (betan > 0 && betam < 0 && betas < 0) {
            return(c(betan, betam, betas))
        }

        betan <- (b - data$lambda * data$alpha * c5) / a * cos(theta) + data$beta[n + 1]
        betam <- (-(b - data$lambda * data$alpha * c5) / a) * sin(theta) + data$beta[m + 1]
        betas <- (data$cSum - data$u[-c(n, m, s)] %*% data$beta[-c(1, n + 1, m + 1, s + 1)] - data$u[n] * betan - data$u[m] * betam) / data$u[s]

        if (betan < 0 && betam > 0 && betas > 0) {
            return(c(betan, betam, betas))
        }

        betan <- (b - data$lambda * data$alpha * c6) / a * cos(theta) + data$beta[n + 1]
        betam <- (-(b - data$lambda * data$alpha * c6) / a) * sin(theta) + data$beta[m + 1]
        betas <- (data$cSum - data$u[-c(n, m, s)] %*% data$beta[-c(1, n + 1, m + 1, s + 1)] - data$u[n] * betan - data$u[m] * betam) / data$u[s]

        if (betan < 0 && betam > 0 && betas < 0) {
            return(c(betan, betam, betas))
        }

        betan <- (b - data$lambda * data$alpha * c7) / a * cos(theta) + data$beta[n + 1]
        betam <- (-(b - data$lambda * data$alpha * c7) / a) * sin(theta) + data$beta[m + 1]
        betas <- (data$cSum - data$u[-c(n, m, s)] %*% data$beta[-c(1, n + 1, m + 1, s + 1)] - data$u[n] * betan - data$u[m] * betam) / data$u[s]

        if (betan < 0 && betam < 0 && betas > 0) {
            return(c(betan, betam, betas))
        }

        betan <- (b - data$lambda * data$alpha * c8) / a * cos(theta) + data$beta[n + 1]
        betam <- (-(b - data$lambda * data$alpha * c8) / a) * sin(theta) + data$beta[m + 1]
        betas <- (data$cSum - data$u[-c(n, m, s)] %*% data$beta[-c(1, n + 1, m + 1, s + 1)] - data$u[n] * betan - data$u[m] * betam) / data$u[s]

        if (betan < 0 && betam < 0 && betas < 0) {
            return(c(betan, betam, betas))
        }

        return(c(NA, NA, NA))
    }

    n <- 5
    m <- 7
    s <- 2
    theta <- 37.32

    R <- zeroSumCdRT(data, n, m, s, theta)

    R <- as.numeric(R)

    C <- .Call("checkMoves", data, as.integer(3), as.integer(n - 1),
        as.integer(m - 1), as.integer(s - 1), as.integer(0),
        PACKAGE = "zeroSum"
    )

    expect_that(R, equals(C, tolerance = 1e-10))
})
