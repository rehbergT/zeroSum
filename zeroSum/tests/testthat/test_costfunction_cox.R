context("Testing cox cost function")

test_that("cox cost function is correct", {
    set.seed(10)
    x <- log2(exampleData$x)
    y <- exampleData$yCox
    P <- ncol(x)
    N <- nrow(x)

    alpha <- 0.5
    lambda <- 0.01
    set.seed(1)
    w <- runif(N)
    v <- runif(P)

    fusion <- Matrix(0, nrow = P - 1, ncol = P, sparse = TRUE)
    for (i in 1:(P - 1)) fusion[i, i] <- 1
    for (i in 1:(P - 1)) fusion[i, (i + 1)] <- -1
    gamma <- 0.30

    cost <- extCostFunction(x, y, rnorm(P + 1), alpha, lambda, family = "cox",
                         gamma = gamma, fusion = fusion, useC = TRUE)

    expect_equal(cost$loglikelihood, cost$C$loglikelihood, tolerance = 1e-10)
    expect_equal(cost$ridge, cost$C$ridge, tolerance = 1e-10)
    expect_equal(cost$lasso, cost$C$lasso, tolerance = 1e-10)
    expect_equal(cost$fusion, cost$C$fusion, tolerance = 1e-10)
    expect_equal(cost$cost, cost$C$cost, tolerance = 1e-10)
})
