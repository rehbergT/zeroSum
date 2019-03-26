context("Testing lambda approximation")

test_that("lambda approximation equals glmnet", {
    x <- log2(exampleData$x)
    y <- exampleData$y

    set.seed(1)
    fi <- sample(rep(rep(1:10), length.out = nrow(x)))
    w <- runif(nrow(x))

    A_zs <- zeroSum(x, y, foldid = fi, weights = w)
    A_gl <- zeroSum(x, y, foldid = fi, weights = w, zeroSum = FALSE)

    ref <- readRDS("references.rds")

    expect_equal(A_zs$lambda[1], ref$test_lambdaApprox$A_zs, tolerance = 1e-8)
    expect_equal(A_gl$lambda[1], ref$test_lambdaApprox$A_gl, tolerance = 1e-8)


    set.seed(1)
    y <- exampleData$ylogistic
    B_zs <- zeroSum(x, y, family = "binomial", foldid = fi, weights = w)
    B_gl <- zeroSum(x, y,
        family = "binomial", foldid = fi, weights = w,
        zeroSum = FALSE
    )

    expect_equal(B_zs$lambda[1], ref$test_lambdaApprox$B_zs, tolerance = 1e-8)
    expect_equal(B_gl$lambda[1], ref$test_lambdaApprox$B_gl, tolerance = 1e-8)
})
