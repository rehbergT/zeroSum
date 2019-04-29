

context("Testing normal coordinate descent move")

test_that("coordinate descent move seems to work", {
    data <- regressionObject(log2(exampleData$x + 1), exampleData$y, "gaussian",
        0.578, 1, 1, NULL,  NULL, NULL, NULL, NULL,  NULL, FALSE, TRUE, FALSE, 
        1, 0.1      
    )

    normalCD <- function(data, k) {
        ak <- data$x[, k]^2 %*% data$w + data$lambda * (1 - data$alpha) * data$v[k]

        bk <- sum(data$x[, k] * data$w * (data$y - rep(data$beta[1], nrow(data$y)) -
            data$x[, -k] %*% data$beta[-c(1, k + 1)]))

        betak <- 0
        bk1 <- bk + data$lambda * data$alpha * data$v[k]
        bk2 <- bk - data$lambda * data$alpha * data$v[k]

        if (bk1 < 0) {
            betak <- bk1 / ak
        } else if (bk2 > 0) {
            betak <- bk2 / ak
        } else {
            betak <- 0
        }

        return(betak)
    }

    k <- 5

    R <- as.numeric(normalCD(data, k))

    C <- .Call("checkMoves", data, as.integer(0),
        as.integer(k - 1), as.integer(0),
        as.integer(0), as.integer(0),
        PACKAGE = "zeroSum"
    )


    expect_that(R, equals(C, tolerance = 1e-10))
})
