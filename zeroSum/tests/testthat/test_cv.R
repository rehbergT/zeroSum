context("Testing cross validation")

test_that("cross validation okay", {

    x <- log2(exampleData$x)
    y <- exampleData$y
    set.seed(1)
    fi <- sample(rep(rep(1:10), length.out = nrow(x)))
    A <- zeroSum(x, y, family = "gaussian", zeroSum = FALSE, foldid = fi,
                 standardize = FALSE, threads = 1)
    B <- zeroSum(x, y, family = "gaussian", foldid = fi,
                 standardize = TRUE,  threads = 4)
    y <- exampleData$ylogistic
    C <- zeroSum(x, y, family = "binomial", zeroSum = FALSE,
                 foldid = fi, standardize = FALSE)
    D <- zeroSum(x, y, family = "binomial", zeroSum = FALSE,
                 foldid = fi, standardize = TRUE)

    y <- exampleData$yCox
    set.seed(1)
    F <- zeroSum(x, y, family = "cox", foldid = fi, zeroSum = FALSE,
                 threads = 1)
    set.seed(1)
    G <- zeroSum(x, y, family = "cox", foldid = fi, zeroSum = FALSE,
                 threads = 4)

    ref <- readRDS("references.rds")
    expect_equal(ref$test_cv$A$cv_stats, A$cv_stats, tolerance = 1e-1)
    expect_equal(ref$test_cv$B$cv_stats, B$cv_stats, tolerance = 1e-1)
    expect_equal(ref$test_cv$C$cv_stats, C$cv_stats, tolerance = 1e-1)
    expect_equal(ref$test_cv$D$cv_stats, D$cv_stats, tolerance = 1e-1)


    expect_equal(F$cv_stats, G$cv_stats, tolerance = 1e-10)
    expect_equal(ref$test_cv$F$cv_stats[1:68, ], F$cv_stats[1:68, ],
                 tolerance = 1e-1)
    expect_equal(ref$test_cv$G$cv_stats[1:68, ], G$cv_stats[1:68, ],
                 tolerance = 1e-1)
    expect_equal(ref$test_cv$glmnetCoefs, as.numeric(coef(F)), tolerance = 1e-1)
})
