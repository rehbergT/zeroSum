context("Testing intercept")

test_that("intercept update seems to work", {
    data <- regressionObject(
        log2(exampleData$x + 1), exampleData$y, "gaussian",
        0.578, 1, 1, NULL, NULL, NULL, NULL, NULL, NULL, FALSE, TRUE, FALSE,
        1, 0.1
    )

    updateOffset <- function(data) {
        a <- (data$w %*% (data$y - data$x %*% data$beta[-1])) / sum(data$w)
        return(a)
    }

    R <- as.numeric(updateOffset(data))
    C <- .Call("checkMoves", data, as.integer(1), as.integer(0),
        as.integer(0), as.integer(0), as.integer(0),
        PACKAGE = "zeroSum"
    )


    expect_that(R, equals(C, tolerance = 1e-10))
})
