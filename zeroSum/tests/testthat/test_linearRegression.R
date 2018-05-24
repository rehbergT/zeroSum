context("Testing linear regression")

test_that( "linear regression equals glmnet",{

    library(glmnet)
    set.seed(10)

    # an example data set is included in the package
    x <- log2(exampleData$x)
    P <- ncol(x)
    N <- nrow(x)
    y <- exampleData$y

    alpha  <- 1.0
    lambda <- 1.32

    ## linear Regression
    A      <- zeroSumFit( x, y, lambda, alpha, algorithm="CD", standardize=FALSE, type="gaussian")
    eA     <- extCostFunction(  x, y, coef(A), lambda, alpha, type="gaussian" )

    A_LS   <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=FALSE, type="gaussian")
    eA_LS  <- extCostFunction(  x, y, coef(A_LS), lambda, alpha, type="gaussian" )

    compA  <- glmnet( x, y, lambda=lambda, alpha=alpha, standardize=FALSE)
    eCompA <- extCostFunction(  x, y, as.numeric(coef(compA)), lambda, alpha, type="gaussian" )

    expect_equal( eA$cost/eCompA$cost, 1.0, tolerance=1e-5)
    expect_equal( eA$cost/eA_LS$cost, 1.0,  tolerance=1e-5)

    expect_equal( cor( as.numeric(coef(A)), as.numeric(coef(A_LS)) ), 1.0, tolerance=1e-5)
    expect_equal( cor( as.numeric(coef(A)), as.numeric(coef(compA)) ), 1.0, tolerance=1e-5)


    ## linear Regression zerosum
    B      <- zeroSumFit( x, y, lambda, alpha, algorithm="CD", standardize=FALSE, type= "gaussianZS")
    eB     <- extCostFunction(  x, y, coef(B), lambda, alpha, type="gaussian" )

    B_LS   <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=FALSE, type= "gaussianZS")
    eB_LS  <- extCostFunction(  x, y, coef(B_LS), lambda, alpha, type="gaussian" )

    expect_equal( eB$cost/eB_LS$cost, 1.0, 1e-5)
    expect_equal( cor( as.numeric(coef(B)), as.numeric(coef(B_LS)) ), 1.0, tolerance=1e-7)

    expect_equal( sum(coef(B)[-1]), 0.0, tolerance=1e-13)
    expect_equal( sum(coef(B_LS)[-1]), 0.0, tolerance=1e-13)


    ## linear Regression standardized
    C      <- zeroSumFit( x, y, lambda, alpha, algorithm="CD", standardize=TRUE, type="gaussian")
    eC     <- extCostFunction(  x, y, coef(C), lambda, alpha, type="gaussian" )

    C_LS   <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=TRUE, type="gaussian")
    eC_LS  <- extCostFunction(  x, y, coef(C_LS), lambda, alpha, type="gaussian"  )

    compC  <- glmnet( x, y, lambda=lambda, alpha=alpha, standardize=TRUE)
    eCompC <- extCostFunction(  x, y, as.numeric(coef(compC)), lambda, alpha, type="gaussian" )

    ## glmnet packages gives here slightly wore results -> higher tolerance
    expect_equal( eC$cost, eCompC$cost, tolerance=1e-3)
    expect_equal( cor( as.numeric(coef(C)), as.numeric(coef(compC)) ), 1.0, tolerance=1e-6)

    ## ls seems to perform a little bit better -> higher tolerance
    expect_equal( eC$cost, eC_LS$cost, tolerance=1e-3)
    expect_equal( cor( as.numeric(coef(C)), as.numeric(coef(C_LS)) ), 1.0, tolerance=1e-6 )


    ## linear Regression zerosum standardized
    D      <- zeroSumFit( x, y, lambda, alpha, algorithm="CD", standardize=TRUE, type= "gaussianZS")
    eD     <- extCostFunction(  x, y, coef(D), lambda, alpha, type="gaussian" )

    D_LS   <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=TRUE, type= "gaussianZS")
    eD_LS  <- extCostFunction(  x, y, coef(D_LS), lambda, alpha, type="gaussian" )

    expect_equal( eD$cost/eD_LS$cost, 1.0, tolerance=1e-3)
    expect_equal( cor( as.numeric(coef(D)), as.numeric(coef(D_LS)) ), 1.0, tolerance=1e-6)

    expect_equal( sum(coef(D)[-1]), 0.0, tolerance=1e-13)
    expect_equal( sum(coef(D_LS)[-1]), 0.0, tolerance=1e-13)


    ## fusion kernel test
    fusion <- Matrix(0, nrow = P-1, ncol = P, sparse = TRUE)
    for(i in 1:(P-1)) fusion[i,i]     <-  1
    for(i in 1:(P-1)) fusion[i,(i+1)] <- -1
    gamma  <- 0.30
    lambda <- 0.01

    FA1  <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=FALSE, gamma=gamma, fusion=fusion, type="fusionGaussian")
    eFA1 <- extCostFunction(  x, y, coef(FA1), lambda, alpha, type="fusionGaussian",gamma=gamma, fusion=fusion )

    FA2  <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=FALSE, gamma=gamma, fusion=fusion, type="fusionGaussianZS")
    eFA2 <- extCostFunction(  x, y, coef(FA2), lambda, alpha, type="fusionGaussian",gamma=gamma, fusion=fusion )

    gamma  <- 0.03
    lambda <- 0.001

    FA3  <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=TRUE, gamma=gamma, fusion=fusion, type="fusionGaussian")
    eFA3 <- extCostFunction(  x, y, coef(FA3), lambda, alpha, type="fusionGaussian",gamma=gamma, fusion=fusion )

    FA4  <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=TRUE, gamma=gamma, fusion=fusion, type="fusionGaussianZS")
    eFA4 <- extCostFunction(  x, y, coef(FA4), lambda, alpha, type="fusionGaussian",gamma=gamma, fusion=fusion )

    costs <- c( eFA1$cost, eFA2$cost, eFA3$cost, eFA4$cost )
    ref_costs <- c(16.0250642115174, 16.5998058865286, 143.31245638638, 93.0005284356821)

    for(i in 1:length(costs))
    expect_lte( costs[i]/ref_costs[i], 1+1e-4)

    # calculate fusion terms and expect that most are adjacent features are equal
    fused <- abs( as.numeric( fusion %*% cbind( coef(FA1), coef(FA2), coef(FA3), coef(FA4) )[-1,] ) )
    expect_gte( sum( fused < 1e-5 ) / length(fused), 0.5)

})
