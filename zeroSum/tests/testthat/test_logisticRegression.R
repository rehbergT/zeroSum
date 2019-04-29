context("Testing logistic regression")

test_that("logistic regression equals glmnet", {
    ref <- readRDS("references.rds")
    x <- log2(exampleData$x)
    y <- exampleData$ylogistic
    set.seed(10)
    alpha <- 0.5
    lambda <- 0.01

    ## logistic Regression
    A <- zeroSum(x, y, alpha, lambda,
        algorithm = "CD", standardize = FALSE,
        family = "binomial", zeroSum = FALSE
    )
    eA <- extCostFunction(x, y, coef(A), alpha, lambda, family = "binomial")

    A_LS <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = FALSE,
        family = "binomial", zeroSum = FALSE
    )
    eA_LS <- extCostFunction(x, y, coef(A_LS), alpha, lambda, family = "binomial")

    A_LS2 <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = FALSE,
        useApprox = FALSE, family = "binomial", zeroSum = FALSE
    )
    eA_LS2 <- extCostFunction(x, y, coef(A_LS2), alpha, lambda,
        family = "binomial"
    )

    eCompA <- extCostFunction(x, y, ref$test_logistic$A, alpha, lambda,
        family = "binomial"
    )

    expect_equal(eA$cost / eA_LS$cost, 1.0, tolerance = 1e-3)
    expect_equal(cor(as.numeric(coef(A)), as.numeric(coef(A_LS))), 1.0,
        tolerance = 1e-3
    )

    expect_equal(eA$cost / eA_LS2$cost, 1.0, tolerance = 1e-3)
    expect_equal(cor(as.numeric(coef(A)), as.numeric(coef(A_LS2))), 1.0,
        tolerance = 1e-3
    )

    expect_equal(eA$cost / eCompA$cost, 1.0, tolerance = 1e-3)
    expect_equal(cor(as.numeric(coef(A)), ref$test_logistic$A), 1.0,
        tolerance = 1e-3
    )


    ## logistic Regression zerosum
    B <- zeroSum(x, y, alpha, lambda,
        algorithm = "CD", standardize = FALSE,
        family = "binomial"
    )
    eB <- extCostFunction(x, y, coef(B), alpha, lambda, family = "binomial")

    B_LS <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = FALSE,
        family = "binomial"
    )
    eB_LS <- extCostFunction(x, y, coef(B_LS), alpha, lambda,
        family = "binomial"
    )

    B_LS2 <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = FALSE,
        useApprox = FALSE, family = "binomial"
    )
    eB_LS2 <- extCostFunction(x, y, coef(B_LS2), alpha, lambda,
        family = "binomial"
    )

    expect_equal(eB$cost, eB_LS$cost, tolerance = 1e-5)
    expect_equal(eB$cost, eB_LS2$cost, tolerance = 1e-5)

    expect_equal(cor(as.numeric(coef(B)), as.numeric(coef(B_LS))), 1.0,
        tolerance = 1e-4
    )
    expect_equal(cor(as.numeric(coef(B_LS)), as.numeric(coef(B_LS2))), 1.0,
        tolerance = 1e-4
    )

    expect_equal(sum(coef(B)[-1]), 0.0, tolerance = 1e-12)
    expect_equal(sum(coef(B_LS)[-1]), 0.0, tolerance = 1e-12)
    expect_equal(sum(coef(B_LS2)[-1]), 0.0, tolerance = 1e-12)

    ## logistic Regression standardized
    C <- zeroSum(x, y, alpha, lambda,
        algorithm = "CD", standardize = TRUE,
        family = "binomial", zeroSum = FALSE
    )
    eC <- extCostFunction(x, y, coef(C), alpha, lambda, family = "binomial")

    C_LS <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = TRUE,
        family = "binomial", zeroSum = FALSE
    )
    eC_LS <- extCostFunction(x, y, coef(C_LS), alpha, lambda, family = "binomial")

    C_LS2 <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = TRUE,
        useApprox = FALSE, family = "binomial", zeroSum = FALSE
    )
    eC_LS2 <- extCostFunction(x, y, coef(C_LS2), alpha, lambda,
        family = "binomial"
    )

    eCompC <- extCostFunction(x, y, ref$test_logistic$C, alpha, lambda,
        family = "binomial"
    )

    ## glmnet packages gives here slightly wore results -> higher tolerance
    expect_equal(eC$cost, eCompC$cost, tolerance = 1e-3)
    expect_equal(cor(as.numeric(coef(C)), ref$test_logistic$C), 1.0,
        tolerance = 1e-4
    )

    ## ls seems to perform a little bit better -> higher tolerance
    expect_equal(eC$cost, eC_LS$cost, tolerance = 1e-3)
    expect_equal(cor(as.numeric(coef(C)), as.numeric(coef(C_LS))), 1.0,
        tolerance = 1e-4
    )

    expect_equal(eC$cost, eC_LS2$cost, tolerance = 1e-3)
    expect_equal(cor(as.numeric(coef(C)), as.numeric(coef(C_LS2))), 1.0,
        tolerance = 1e-4
    )


    ## logistic Regression zerosum standardized
    D <- zeroSum(x, y, alpha, lambda,
        algorithm = "CD", standardize = TRUE,
        family = "binomial"
    )
    eD <- extCostFunction(x, y, coef(D), alpha, lambda, family = "binomial")

    D_LS <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = TRUE,
        family = "binomial"
    )
    eD_LS <- extCostFunction(x, y, coef(D_LS), alpha, lambda,
        family = "binomial"
    )

    D_LS2 <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = TRUE,
        useApprox = FALSE, family = "binomial"
    )
    eD_LS2 <- extCostFunction(x, y, coef(D_LS2), alpha, lambda,
        family = "binomial"
    )

    expect_equal(eD$cost / eD_LS$cost, 1.0, tolerance = 1e-2)
    expect_equal(cor(as.numeric(coef(D)), as.numeric(coef(D_LS))), 1.0,
        tolerance = 1e-3
    )

    expect_equal(eD$cost / eD_LS2$cost, 1.0, tolerance = 1e-2)
    expect_equal(cor(as.numeric(coef(D)), as.numeric(coef(D_LS2))), 1.0,
        tolerance = 1e-3
    )

    expect_equal(sum(coef(D)[-1]), 0.0, tolerance = 1e-14)
    expect_equal(sum(coef(D_LS)[-1]), 0.0, tolerance = 1e-14)
    expect_equal(sum(coef(D_LS2)[-1]), 0.0, tolerance = 1e-14)


    ## fusion kernel test
    P <- ncol(x)
    fusion <- Matrix(0, nrow = P - 1, ncol = P, sparse = TRUE)
    for (i in 1:(P - 1)) fusion[i, i] <- 1
    for (i in 1:(P - 1)) fusion[i, (i + 1)] <- -1
    gamma <- 0.03
    lambda <- 0.001
    set.seed(1)

    FA1 <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = FALSE,
        gamma = gamma, fusion = fusion, family = "binomial",
        zeroSum = FALSE
    )
    eFA1 <- extCostFunction(x, y, coef(FA1), alpha, lambda,
        family = "binomial",
        gamma = gamma, fusion = fusion
    )

    FA2 <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = FALSE,
        gamma = gamma, fusion = fusion, useApprox = FALSE,
        family = "binomial", zeroSum = FALSE
    )
    eFA2 <- extCostFunction(x, y, coef(FA2), alpha, lambda,
        family = "binomial",
        gamma = gamma, fusion = fusion
    )

    FA3 <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = FALSE,
        gamma = gamma, fusion = fusion, family = "binomial"
    )
    eFA3 <- extCostFunction(x, y, coef(FA3), alpha, lambda,
        family = "binomial",
        gamma = gamma, fusion = fusion
    )

    FA4 <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = FALSE,
        gamma = gamma, fusion = fusion, useApprox = FALSE,
        family = "binomial"
    )
    eFA4 <- extCostFunction(x, y, coef(FA4), alpha, lambda,
        family = "binomial",
        gamma = gamma, fusion = fusion
    )

    FA5 <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = TRUE,
        gamma = gamma, fusion = fusion, family = "binomial",
        zeroSum = FALSE
    )
    eFA5 <- extCostFunction(x, y, coef(FA5), alpha, lambda,
        family = "binomial",
        gamma = gamma, fusion = fusion
    )

    FA6 <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = TRUE,
        gamma = gamma, fusion = fusion, useApprox = FALSE,
        family = "binomial", zeroSum = FALSE
    )
    eFA6 <- extCostFunction(x, y, coef(FA6), alpha, lambda,
        family = "binomial",
        gamma = gamma, fusion = fusion
    )

    FA7 <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = TRUE,
        gamma = gamma, fusion = fusion, family = "binomial"
    )
    eFA7 <- extCostFunction(x, y, coef(FA7), alpha, lambda,
        family = "binomial",
        gamma = gamma, fusion = fusion
    )

    FA8 <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = TRUE,
        gamma = gamma, fusion = fusion, useApprox = FALSE,
        family = "binomial"
    )
    eFA8 <- extCostFunction(x, y, coef(FA8), alpha, lambda,
        family = "binomial",
        gamma = gamma, fusion = fusion
    )

    costs <- c(
        eFA1$cost, eFA2$cost, eFA3$cost, eFA4$cost, eFA5$cost, eFA6$cost,
        eFA7$cost, eFA8$cost
    )

    for (i in 1:length(costs))
        expect_lte(costs[i], ref$test_logistic$fusionCosts[i] + 5e-2)

    # calculate fusion terms and expect that most adjacent features are equal
    fused <- abs(as.numeric(fusion %*% cbind(coef(FA1), coef(FA2), coef(FA3), coef(FA4), coef(FA5), coef(FA6), coef(FA7), coef(FA8))[-1, ]))
    expect_gte(sum(fused < 1e-5) / length(fused), 0.3)
})
