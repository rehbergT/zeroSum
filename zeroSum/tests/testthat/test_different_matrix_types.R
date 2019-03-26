context("Testing converting of different matrix types")

test_that("Casting different matrix types to numeric matrix", {
    set.seed(1)

    x <- log2(exampleData$x)
    y <- exampleData$y

    x2 <- data.frame(x)
    y2 <- data.frame(y)

    fi <- sample(rep(rep(1:10), length.out = nrow(x)))
    A <- zeroSum(x, y, family = "gaussian", foldid = fi)
    B <- zeroSum(x2, y2, family = "gaussian", foldid = fi)

    expect_equal(A$cv_stats, B$cv_stats, tolerance = 1e-10)
})
