context("Testing lambda approximation")

test_that( "lambda approximation equals glmnet",{

    library(glmnet)

    x <- log2(exampleData$x)
    y <- exampleData$y

    set.seed(1)
    fi <- c(9L, 2L, 7L, 6L, 6L, 4L, 8L, 6L, 5L, 2L, 5L, 4L, 4L, 7L, 9L,
            8L, 1L, 9L, 1L, 10L, 3L, 3L, 7L, 2L, 5L, 10L, 1L, 8L, 1L, 3L,
            10L)

    sampleWeights <- c(0.0320834515121008, 0.000489948537686952, 0.0160877017604945,
            0.0151914328774442, 0.0445583053697056, 0.014263259516389, 0.0396746828799479,
            0.049625381470771, 0.0519775863118428, 0.00400601880476022, 0.04133248041794,
            0.0156638481947184, 0.0054797894103371, 0.0522529930797012, 0.0227622121752009,
            0.0249253136672002, 0.0531833404448112, 0.0319841930255701, 0.0526985820061141,
            0.0417173598418207, 0.0391326186629912, 0.0545830720743945, 0.0277277385328431,
            0.0268334959752057, 0.035553647062309, 0.0455020883670454, 0.0263984014519289,
            0.0461012463125131, 0.028134715137047, 0.0290151639152878, 0.0310599312038777)

    A <- zeroSumCVFit( x, y, type="gaussian", foldid=fi, weights=sampleWeights )
    B <- zeroSumCVFit( x, y, type="gaussianZS", foldid=fi, weights=sampleWeights )
    C <- cv.glmnet( x, y, standardize=FALSE, foldid=fi, weights=sampleWeights )

    zeroSumLambdaMax <- 42.546947791468

    expect_that( A$lambda[1], equals( C$lambda[1], tolerance=1e-10) )
    expect_that( B$lambda[1], equals( zeroSumLambdaMax, tolerance=1e-10) )


    set.seed(1)
    y <- exampleData$ylogistic
    A <- zeroSumCVFit( x, y, type="binomial", foldid=fi, weights=sampleWeights )
    B <- zeroSumCVFit( x, y, type="binomialZS", foldid=fi, weights=sampleWeights )
    C <- cv.glmnet( x, y, standardize=FALSE, foldid=fi, weights=sampleWeights, family="binomial" )

    zeroSumLambdaMax <- 0.75930420730324

    expect_that( A$lambda[1], equals( C$lambda[1], tolerance=1e-10) )
    expect_that( B$lambda[1], equals( zeroSumLambdaMax, tolerance=1e-10) )

})
