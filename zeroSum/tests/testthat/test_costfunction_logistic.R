context("Testing logistic cost function")

test_that( "check whether the R and C logistic cost function yield the same result",{

    library(glmnet)
    set.seed(10)

    ## logistic regression test
    x <- log2(exampleData$x)
    P <- ncol(x)
    N <- nrow(x)
    y <- exampleData$ylogistic

    alpha  <- 0.5
    lambda <- 0.01
    w <- runif(N)
    v <- runif(P)

    lin  <- zeroSumFit( x, y, lambda, alpha, weights=w, type="binomial")

    fusion <- Matrix(0, nrow = P-1, ncol = P, sparse = TRUE)
    for(i in 1:(P-1)) fusion[i,i]     <-  1
    for(i in 1:(P-1)) fusion[i,(i+1)] <- -1
    gamma  <- 0.30

    cost <- extCostFunction(  x, y, coef(lin), lambda, alpha, type="fusionBinomial",gamma=gamma, fusion=fusion, useC=TRUE )

    expect_equal( cost$loglikelihood, cost$C$loglikelihood, tolerance=1e-15)
    expect_equal( cost$ridge,         cost$C$ridge,         tolerance=1e-15)
    expect_equal( cost$lasso,         cost$C$lasso,         tolerance=1e-15)
    expect_equal( cost$fusion,        cost$C$fusion,        tolerance=1e-15)
    expect_equal( cost$cost,          cost$C$cost,          tolerance=1e-15)

})
