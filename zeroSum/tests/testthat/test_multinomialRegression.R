context("Testing multinomial regression")

test_that( "multinomial regression equals glmnet",{

    library(glmnet)
    set.seed(10)

    data(MultinomialExample)
    P <- ncol(x)
    x <- x[1:50,]
    y <- y[1:50]
    N <- nrow(x)
    K <- max(y)

    alpha  <- 1.0
    lambda <- 0.01

    ## multinomial Regression
    A      <- zeroSumFit( x, y, lambda, alpha, algorithm="CD", standardize=FALSE, verbose=FALSE, type="multinomial")
    eA     <- extCostFunction(  x, y, coef(A), lambda, alpha, type="multinomial")

    A_LS   <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=FALSE, useApprox=TRUE, verbose=FALSE, type="multinomial")
    eA_LS  <- extCostFunction(  x, y, coef(A_LS), lambda, alpha, type="multinomial")

    A_LS2   <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=FALSE, useApprox=FALSE, verbose=FALSE, type="multinomial")
    eA_LS2  <- extCostFunction(  x, y, coef(A_LS2), lambda, alpha, type="multinomial")

    compA  <- glmnet( x, y, lambda=lambda, alpha=alpha, standardize=FALSE, family="multinomial")
    beta   <- coef(compA)
    betaComb <- matrix(0,nrow=(P+1), ncol=K)
    for( i in 1:K ) betaComb[,i] <- as.numeric( beta[[i]])

    eCompA <- extCostFunction( x, y, betaComb, lambda, alpha, type="multinomial" )


    expect_equal( eA$cost, eA_LS$cost, tolerance=1e-5)
    expect_equal( cor( as.numeric(coef(A)), as.numeric(coef(A_LS)) ),  1.0, tolerance=1e-5)

    expect_equal( eA$cost, eA_LS2$cost, tolerance=1e-5)
    expect_equal( cor( as.numeric(coef(A)), as.numeric(coef(A_LS2)) ), 1.0, tolerance=1e-5)

    expect_equal( eA$cost, eCompA$cost, tolerance=1e-5)
    expect_equal( cor( as.numeric(coef(A)), as.numeric(betaComb)), 1.0, tolerance=1e-2)


    # ## multinomial Regression zerosum
    B      <- zeroSumFit( x, y, lambda, alpha, algorithm="CD", standardize=FALSE, verbose=FALSE, type="multinomialZS", precision=1e-10)
    eB     <- extCostFunction(  x, y, coef(B), lambda, alpha, type="multinomial")

    B_LS   <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=FALSE, verbose=FALSE, type="multinomialZS", precision=1e-10)
    eB_LS  <- extCostFunction(  x, y, coef(B_LS), lambda, alpha, type="multinomial")

    B_LS2   <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=FALSE, useApprox=FALSE, verbose=FALSE, type="multinomialZS", precision=1e-10)
    eB_LS2  <- extCostFunction(  x, y, coef(B_LS2), lambda, alpha, type="multinomial")

    expect_lte( eB$cost,     0.333521)
    expect_lte( eB_LS$cost,  0.335500)
    expect_lte( eB_LS2$cost, 0.335800)

    expect_equal( cor( as.numeric(coef(B)), as.numeric(coef(B_LS)) ), 1.0, tolerance=1e-2)
    expect_equal( cor( as.numeric(coef(B_LS)), as.numeric(coef(B_LS2)) ), 1.0, tolerance=1e-2)

    expect_equal( sum(coef(B)[-1,1]), sum(coef(B)[-1,2]), tolerance=1e-12)
    expect_equal( sum(coef(B)[-1,1]), sum(coef(B)[-1,3]), tolerance=1e-12)

    expect_equal( sum(coef(B_LS)[-1,1]), sum(coef(B_LS)[-1,2]), tolerance=1e-12)
    expect_equal( sum(coef(B_LS)[-1,1]), sum(coef(B_LS)[-1,3]), tolerance=1e-12)

    expect_equal( sum(coef(B_LS2)[-1,1]), sum(coef(B_LS2)[-1,2]), tolerance=1e-12)
    expect_equal( sum(coef(B_LS2)[-1,1]), sum(coef(B_LS2)[-1,3]), tolerance=1e-12)

    ## multinomial Regression standardized
    C      <- zeroSumFit( x, y, lambda, alpha, algorithm="CD", standardize=TRUE, verbose=FALSE, type="multinomial")
    eC     <- extCostFunction(  x, y, coef(C), lambda, alpha, type="multinomial")

    C_LS   <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=TRUE, verbose=FALSE, type="multinomial")
    eC_LS  <- extCostFunction(  x, y, coef(C_LS), lambda, alpha, type="multinomial")

    C_LS2  <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=TRUE, useApprox=FALSE, verbose=FALSE, type="multinomial")
    eC_LS2 <- extCostFunction(  x, y, coef(C_LS2), lambda, alpha, type="multinomial")

    compC  <- glmnet( x, y, lambda=lambda, alpha=alpha, standardize=TRUE, family="multinomial")
    beta   <- coef(compA)
    betaComb <- matrix(0,nrow=(P+1), ncol=K)
    for( i in 1:K ) betaComb[,i] <- as.numeric( beta[[i]])

    eCompC <- extCostFunction( x, y, betaComb, lambda, alpha, type="multinomial" )

    ## glmnet packages gives here slightly wore results -> higher tolerance
    expect_equal( eC$cost, eCompC$cost, tolerance=1e-3)
    expect_equal( cor( as.numeric(coef(C)), as.numeric(betaComb) ), 1.0, tolerance=1e-2)

    ## ls seems to perform a little bit better -> higher tolerance
    expect_equal( eC$cost, eC_LS$cost, tolerance=1e-3)
    expect_equal( cor( as.numeric(coef(C)), as.numeric(coef(C_LS)) ), 1.0, tolerance=1e-4)

    expect_equal( eC$cost, eC_LS2$cost, tolerance=1e-3)
    expect_equal( cor( as.numeric(coef(C)), as.numeric(coef(C_LS2)) ), 1.0, tolerance=1e-4)

    ## multinomial Regression zerosum standardized
    D      <- zeroSumFit( x, y, lambda, alpha, algorithm="CD", standardize=TRUE, verbose=FALSE, type= "multinomialZS")
    eD     <- extCostFunction(  x, y, coef(D), lambda, alpha, type="multinomial")

    D_LS   <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=TRUE, verbose=FALSE, type= "multinomialZS")
    eD_LS  <- extCostFunction(  x, y, coef(D_LS), lambda, alpha, type="multinomial")

    D_LS2  <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=TRUE, useApprox=FALSE, verbose=FALSE, type= "multinomialZS")
    eD_LS2 <- extCostFunction(  x, y, coef(D_LS2), lambda, alpha, type="multinomial")

    expect_lte( eD$cost,     0.3345)
    expect_lte( eD_LS$cost,  0.3365)
    expect_lte( eD_LS2$cost, 0.3365)

    expect_equal( cor( as.numeric(coef(D)), as.numeric(coef(D_LS)) ), 1.0, tolerance=1e-2)
    expect_equal( cor( as.numeric(coef(D_LS)), as.numeric(coef(D_LS2)) ), 1.0, tolerance=1e-2)

    expect_equal( sum(coef(D)[-1,1]), sum(coef(D)[-1,2]), tolerance=1e-12)
    expect_equal( sum(coef(D)[-1,1]), sum(coef(D)[-1,3]), tolerance=1e-12)

    expect_equal( sum(coef(D_LS)[-1,1]), sum(coef(D_LS)[-1,2]), tolerance=1e-12)
    expect_equal( sum(coef(D_LS)[-1,1]), sum(coef(D_LS)[-1,3]), tolerance=1e-12)

    expect_equal( sum(coef(D_LS2)[-1,1]), sum(coef(D_LS2)[-1,2]), tolerance=1e-12)
    expect_equal( sum(coef(D_LS2)[-1,1]), sum(coef(D_LS2)[-1,3]), tolerance=1e-12)


    ## fusion kernel test
    fusion <- Matrix(0, nrow = P-1, ncol = P, sparse = TRUE)
    for(i in 1:(P-1)) fusion[i,i]     <-  1
    for(i in 1:(P-1)) fusion[i,(i+1)] <- -1
    gamma  <- 0.03
    lambda <- 0.001

    FA1  <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=FALSE, gamma=gamma, fusion=fusion, verbose=TRUE, type="fusionMultinomial")
    eFA1 <- extCostFunction(  x, y, coef(FA1), lambda, alpha, type="fusionMultinomial",gamma=gamma, fusion=fusion)

    FA2  <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=FALSE, gamma=gamma, fusion=fusion, useApprox=FALSE, verbose=TRUE, type="fusionMultinomial")
    eFA2 <- extCostFunction(  x, y, coef(FA2), lambda, alpha, type="fusionMultinomial",gamma=gamma, fusion=fusion)

    FA3  <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=FALSE, gamma=gamma, fusion=fusion, verbose=TRUE, type="fusionMultinomialZS")
    eFA3 <- extCostFunction(  x, y, coef(FA3), lambda, alpha, type="fusionMultinomial",gamma=gamma, fusion=fusion)

    FA4  <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=FALSE, gamma=gamma, fusion=fusion, useApprox=FALSE, verbose=TRUE, type="fusionMultinomialZS")
    eFA4 <- extCostFunction(  x, y, coef(FA4), lambda, alpha, type="fusionMultinomial",gamma=gamma, fusion=fusion)

    FA5  <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=TRUE, gamma=gamma, fusion=fusion, verbose=TRUE, type="fusionMultinomial")
    eFA5 <- extCostFunction(  x, y, coef(FA5), lambda, alpha, type="fusionMultinomial",gamma=gamma, fusion=fusion)

    FA6  <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=TRUE, gamma=gamma, fusion=fusion, useApprox=FALSE, verbose=TRUE, type="fusionMultinomial")
    eFA6 <- extCostFunction(  x, y, coef(FA6), lambda, alpha, type="fusionMultinomial",gamma=gamma, fusion=fusion)

    FA7  <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=TRUE, gamma=gamma, fusion=fusion, verbose=TRUE, type="fusionMultinomialZS")
    eFA7 <- extCostFunction(  x, y, coef(FA7), lambda, alpha, type="fusionMultinomial",gamma=gamma, fusion=fusion)

    FA8  <- zeroSumFit( x, y, lambda, alpha, algorithm="LS", standardize=TRUE, gamma=gamma, fusion=fusion, useApprox=FALSE, verbose=TRUE, type="fusionMultinomialZS")
    eFA8 <- extCostFunction(  x, y, coef(FA8), lambda, alpha, type="fusionMultinomial",gamma=gamma, fusion=fusion)

    costs <- c( eFA1$cost, eFA2$cost, eFA3$cost, eFA4$cost, eFA5$cost, eFA6$cost, eFA7$cost, eFA8$cost)
    ref_costs <- c(0.73507406833032, 0.737415428458164, 0.733279858131882, 0.728726039716532,
                    0.72439005896725, 0.727797313867489, 0.730339728819096, 0.733697299636398)




    for(i in 1:length(costs))
    expect_lte( costs[i]/ref_costs[i], 1+1e-3)

    # calculate fusion terms and expect that most are adjacent features are equal
    fused <- abs( as.numeric( fusion %*% cbind( coef(FA1), coef(FA2), coef(FA3), coef(FA4) )[-1,] ) )
    expect_gte( sum( fused < 1e-5 ) / length(fused), 0.5)

})
