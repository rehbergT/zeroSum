context("Testing rotated moves")

test_that("rotated move is correct", {

    x <- log2(exampleData$x)
    for (i in 1:2) x <- cbind(x,x)
    y <- exampleData$y

    set.seed(1)
    fi <- sample( rep( rep(1:5), length.out=nrow(x) ))

    A <- zeroSum( x, y, foldid=fi, rotatedUpdates=TRUE )
    B <- zeroSum( x, y, foldid=fi )

    for (i in 1:nrow(A$cv_stats)) {
        expect_equal(as.numeric(A$cv_stats[i, 3]) / as.numeric(B$cv_stats[i, 3]),
                     1.0, tolerance = 1e-5)
        expect_equal(as.numeric(A$cv_stats[i, 4]) / as.numeric(B$cv_stats[i, 4]),
                     1.0, tolerance = 1e-3)
    }

})