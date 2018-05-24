context("Testing cox regression")

test_that( "cox regression equals glmnet",{

    lsCdDiff    <- 1e-4  ## allowed difference between local search and coordinate descent results
    zeroSumDiff <- 1e-14 ## allowed deviation from zero contraint
    glmnetDiff  <- 1e-4  ## allowed difference between glmnet and zeroSum results

    library(glmnet)
    set.seed(10)

    data(CoxExample)

    P <- ncol(x)
    N <- nrow(x)

    set.seed(4)
    w <- runif(nrow(x))

    alpha  <- 1.0
    lambda <- 0.2

    A      <- zeroSumFit( x, y, lambda, alpha, type = "cox", standardize=FALSE, algorithm="CD", weights=w, verbose=TRUE)
    eA     <- extCostFunction(  x, y, coef(A), lambda, alpha=alpha, type="cox" )

    A_LS   <- zeroSumFit( x, y, lambda, alpha, type = "cox", standardize=FALSE, algorithm="LS", weights=w, verbose=TRUE)
    eA_LS  <- extCostFunction(  x, y, coef(A_LS), lambda, alpha=1, type="cox" )

    A_LS2  <- zeroSumFit( x, y, lambda, alpha, type = "cox", standardize=FALSE, algorithm="LS", useApprox=FALSE, weights=w, verbose=TRUE)
    eA_LS2 <- extCostFunction(  x, y, coef(A_LS2), lambda, alpha=1, type="cox" )

    compA  <- glmnet( x, y, lambda=lambda, alpha=alpha, family = "cox", standardize=FALSE, weights=w)
    eCompA <- extCostFunction(  x, y, c(0.0,as.numeric(coef(compA))), lambda, alpha=1, type="cox" )

    expect_equal( eA$cost, eA_LS$cost, tolerance=lsCdDiff)
    expect_equal( cor( as.numeric(coef(A)), as.numeric(coef(A_LS)) ),  1.0, tolerance=1e-3)

    expect_equal( eA$cost, eA_LS2$cost, tolerance=lsCdDiff)
    expect_equal( cor( as.numeric(coef(A)), as.numeric(coef(A_LS2)) ), 1.0, tolerance=1e-2)

    expect_equal( eA$cost, eCompA$cost, tolerance=lsCdDiff)
    expect_equal( cor( as.numeric(coef(A)), c(0.0,as.numeric(coef(compA))) ), 1.0, tolerance=1e-3)



    B      <- zeroSumFit( x, y, lambda, alpha, algorithm="CD", standardize=FALSE, verbose=TRUE, type= "coxZS" )
    eB     <- extCostFunction(  x, y, coef(B), lambda, alpha, type="cox" )

    B_LS   <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=FALSE, verbose=TRUE, type= "coxZS" )
    eB_LS  <- extCostFunction(  x, y, coef(B_LS), lambda, alpha, type="cox" )

    B_LS2   <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=FALSE, useApprox=FALSE, verbose=TRUE, type= "coxZS")
    eB_LS2  <- extCostFunction(  x, y, coef(B_LS2), lambda, alpha, type="cox" )


    expect_equal( eB$cost, eB_LS$cost, tolerance=lsCdDiff)
    expect_equal( eB$cost, eB_LS2$cost, tolerance=lsCdDiff)

    expect_equal( cor( as.numeric(coef(B)), as.numeric(coef(B_LS)) ), 1.0, tolerance=1e-4)
    expect_equal( cor( as.numeric(coef(B_LS)), as.numeric(coef(B_LS2)) ), 1.0, tolerance=1e-6)

    expect_equal( sum(coef(B)[-1]), 0.0, tolerance=zeroSumDiff)
    expect_equal( sum(coef(B_LS)[-1]), 0.0, tolerance=zeroSumDiff)



    C      <- zeroSumFit( x, y, lambda, alpha, algorithm="CD", standardize=TRUE, verbose=TRUE, type="cox" )
    eC     <- extCostFunction(  x, y, coef(C), lambda, alpha, type="cox" )

    C_LS   <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=TRUE, verbose=TRUE, type="cox" )
    eC_LS  <- extCostFunction(  x, y, coef(C_LS), lambda, alpha, type="cox" )

    C_LS2   <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=TRUE, useApprox=FALSE, verbose=TRUE, type="cox" )
    eC_LS2  <- extCostFunction(  x, y, coef(C_LS2), lambda, alpha, type="cox" )

    compC  <- glmnet( x, y, lambda=lambda, alpha=alpha, standardize=TRUE, family="cox" )
    eCompC <- extCostFunction(  x, y, c(0.0,as.numeric(coef(compC))), lambda, alpha, type="cox" )

    ## glmnet packages gives here slightly wore results -> higher tolerance
    expect_equal( eC$cost, eCompC$cost, tolerance=1e-3)
    expect_equal( cor( as.numeric(coef(C)), c(0.0,as.numeric(coef(compC))) ), 1.0, tolerance=1e-3)

    ## ls seems to perform a little bit better -> higher tolerance
    expect_equal( eC$cost, eC_LS$cost, tolerance=1e-3)
    expect_equal( cor( as.numeric(coef(C)), as.numeric(coef(C_LS)) ), 1.0, tolerance=1e-3)

    expect_equal( eC$cost, eC_LS2$cost, tolerance=1e-3)
    expect_equal( cor( as.numeric(coef(C)), as.numeric(coef(C_LS2)) ), 1.0, tolerance=1e-2)


    D      <- zeroSumFit( x, y, lambda, alpha, algorithm="CD", standardize=TRUE, verbose=TRUE, type= "coxZS")
    eD     <- extCostFunction(  x, y, coef(D), lambda, alpha, type="cox" )

    D_LS   <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=TRUE, verbose=TRUE, type= "coxZS")
    eD_LS  <- extCostFunction(  x, y, coef(D_LS), lambda, alpha, type="cox" )

    D_LS2   <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=TRUE, useApprox=FALSE, verbose=TRUE, type= "coxZS")
    eD_LS2  <- extCostFunction(  x, y, coef(D_LS2), lambda, alpha, type="cox" )

    expect_equal( eD$cost/eD_LS$cost, 1.0, tolerance=1e-3)
    expect_equal( cor( as.numeric(coef(D)), as.numeric(coef(D_LS)) ), 1.0, tolerance=1e-3)

    expect_equal( eD$cost/eD_LS2$cost, 1.0, tolerance=1e-3)
    expect_equal( cor( as.numeric(coef(D)), as.numeric(coef(D_LS2)) ), 1.0, tolerance=lsCdDiff)

    expect_equal( sum(coef(D)[-1]), 0.0, tolerance=zeroSumDiff)
    expect_equal( sum(coef(D_LS)[-1]), 0.0, tolerance=zeroSumDiff)
    expect_equal( sum(coef(D_LS2)[-1]), 0.0, tolerance=zeroSumDiff)


    ## fusion kernel test
    fusion <- Matrix(0, nrow = P-1, ncol = P, sparse = TRUE)
    for(i in 1:(P-1)) fusion[i,i]     <-  1
    for(i in 1:(P-1)) fusion[i,(i+1)] <- -1
    gamma  <- 0.03
    lambda <- 0.001

    FA1  <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=FALSE, gamma=gamma, fusion=fusion, verbose=TRUE, type="fusionCox")
    eFA1 <- extCostFunction(  x, y, coef(FA1), lambda, alpha, type="fusionCox",gamma=gamma, fusion=fusion )

    FA2  <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=FALSE, gamma=gamma, fusion=fusion, useApprox=FALSE, verbose=TRUE, type="fusionCox")
    eFA2 <- extCostFunction(  x, y, coef(FA2), lambda, alpha, type="fusionCox",gamma=gamma, fusion=fusion )

    FA3  <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=FALSE, gamma=gamma, fusion=fusion, verbose=TRUE, type="fusionCoxZS")
    eFA3 <- extCostFunction(  x, y, coef(FA3), lambda, alpha, type="fusionCox",gamma=gamma, fusion=fusion )

    FA4  <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=FALSE, gamma=gamma, fusion=fusion, useApprox=FALSE, verbose=TRUE, type="fusionCoxZS")
    eFA4 <- extCostFunction(  x, y, coef(FA4), lambda, alpha, type="fusionCox",gamma=gamma, fusion=fusion )

    FA5  <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=TRUE, gamma=gamma, fusion=fusion, verbose=TRUE, type="fusionCox")
    eFA5 <- extCostFunction(  x, y, coef(FA5), lambda, alpha, type="fusionCox",gamma=gamma, fusion=fusion )

    FA6  <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=TRUE, gamma=gamma, fusion=fusion, useApprox=FALSE, verbose=TRUE, type="fusionCox")
    eFA6 <- extCostFunction(  x, y, coef(FA6), lambda, alpha, type="fusionCox",gamma=gamma, fusion=fusion )

    FA7  <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=TRUE, gamma=gamma, fusion=fusion, verbose=TRUE, type="fusionCoxZS")
    eFA7 <- extCostFunction(  x, y, coef(FA7), lambda, alpha, type="fusionCox",gamma=gamma, fusion=fusion )

    FA8  <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=TRUE, gamma=gamma, fusion=fusion, useApprox=FALSE, verbose=TRUE, type="fusionCoxZS")
    eFA8 <- extCostFunction(  x, y, coef(FA8), lambda, alpha, type="fusionCox",gamma=gamma, fusion=fusion )


    costs <- c( eFA1$cost, eFA2$cost, eFA3$cost, eFA4$cost, eFA5$cost, eFA6$cost, eFA7$cost, eFA8$cost)
    ref_costs <- c(-0.826632848822714, -0.826500937115942, -0.82463953869077,
                    -0.824333287261024, -0.82654496158336, -0.82658642554583, -0.824168754877771,
                    -0.8245957081152)

    for(i in 1:length(costs))
    expect_lte( costs[i]/ref_costs[i], 1+1e-3)

    # calculate fusion terms and expect that most are adjacent features are equal
    fused <- abs( as.numeric( fusion %*% cbind( coef(FA1), coef(FA2), coef(FA3), coef(FA4) )[-1,] ) )
    expect_gte( sum( fused < 1e-5 ) / length(fused), 0.5)


})
