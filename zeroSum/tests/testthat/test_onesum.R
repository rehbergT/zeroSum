context("Testing one sum")

test_that("one sum okay", {

    x <- log2(exampleData$x)
    set.seed(1)
    fi <- sample(rep(rep(1:10), length.out = nrow(x)))

    y <- exampleData$y
    A <- zeroSum(x, y, family = "gaussian", foldid = fi, cSum = 1)

    y <- exampleData$ylogistic
    B <- zeroSum(x, y, family = "binomial", foldid = fi,  cSum = 1)

    y <- exampleData$yCox
    D <- zeroSum(x, y, family = "cox", foldid = fi, cSum = 1)

    expect_equal(sum(coef(A)[-1, ]), 1.0, tolerance = 1e-10)
    expect_equal(sum(coef(B)[-1, ]), 1.0, tolerance = 1e-10)
    expect_equal(sum(coef(D)[-1, ]), 1.0, tolerance = 1e-10)

})
