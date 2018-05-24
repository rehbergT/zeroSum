context("Testing regression weights")

test_that( "regression weights seem to be okay",{

    set.seed(10)

    # an example data set is included in the package
    x <- log2(exampleData$x)
    y <- exampleData$y

    P <- ncol(x)
    N <- nrow(x)

    alpha  <- 1.0

    w <- runif(N)
    u <- rep(1,P)
    v <- rep(1,P)

    v[1] <- 0
    u[1] <- 0
    A <- zeroSumCVFit( x, y, alpha, weights=w, zeroSumWeights=u, penalty.factor=v, cores=11 )

    expect_equal( sum( coef(A)[-c(1,2),] ), 0, tolerance=1e-12 )
    expect_equal( as.numeric(coef(A)[2,1]), -2.24788525113661, tolerance=1e-2 )

    y <- exampleData$ylogistic
    lambda <- 0.01
    u <- rep(1,P)
    v <- rep(1,P)

    v[1]  <- 0
    v[19] <- 0
    u[1]  <- 0
    u[19] <- 0

    fit1 <- zeroSumFit( x, y, lambda=lambda, alpha, type="binomialZS", penalty.factor=v, zeroSumWeights=u )
    fit2 <- zeroSumFit( x, y, lambda=lambda, alpha, type="binomialZS", algorithm="LS", penalty.factor=v, zeroSumWeights=u )

    expect_equal( sum( coef(fit1)[-c(1,2,20),] ), 0, tolerance=1e-12 )
    expect_equal( sum( coef(fit2)[-c(1,2,20),] ), 0, tolerance=1e-12 )


    e1 <- extCostFunction(  x, y, as.numeric(coef(fit1)), lambda, alpha, type="binomial", penalty.factor=v )
    e2 <- extCostFunction(  x, y, as.numeric(coef(fit2)), lambda, alpha, type="binomial", penalty.factor=v )

    cbind(e1,e2)
    diff <- as.numeric(e1) - as.numeric(e2)
    expect_lte( diff[1], 1e-2)
    expect_lte( diff[3], 1e-2)
    expect_lte( diff[4], 1e-2)

})
