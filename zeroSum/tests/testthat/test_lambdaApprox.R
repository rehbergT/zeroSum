context("Testing lambda approximation")

test_that("lambda approximation equals glmnet", {
    x <- log2(exampleData$x)
    y <- exampleData$y

    set.seed(1)
    fi <- c(
        9L, 2L, 7L, 6L, 6L, 4L, 8L, 6L, 5L, 2L, 5L, 4L, 4L, 7L, 9L,
        8L, 1L, 9L, 1L, 10L, 3L, 3L, 7L, 2L, 5L, 10L, 1L, 8L, 1L, 3L,
        10L
    )
    w <- c(
        0.599565825425088, 0.493541307048872, 0.186217601411045, 0.827373318606988,
        0.668466738192365, 0.79423986072652, 0.107943625887856, 0.723710946040228,
        0.411274429643527, 0.820946294115856, 0.647060193819925, 0.78293276228942,
        0.553036311641335, 0.529719580197707, 0.789356231689453, 0.023331202333793,
        0.477230065036565, 0.7323137386702, 0.692731556482613, 0.477619622135535,
        0.8612094768323, 0.438097107224166, 0.244797277031466, 0.0706790471449494,
        0.0994661601725966, 0.31627170718275, 0.518634263193235, 0.662005076417699,
        0.406830187188461, 0.912875924259424, 0.293603372760117
    )

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
