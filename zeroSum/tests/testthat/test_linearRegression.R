context("Testing linear regression")

test_that("linear regression equals glmnet", {
    ref <- readRDS("references.rds")
    x <- log2(exampleData$x)
    y <- exampleData$y
    set.seed(10)
    alpha <- 0.4
    lambda <- 1.32

    ## linear Regression
    A <- zeroSum(x, y, alpha, lambda,
        algorithm = "CD", standardize = FALSE,
        family = "gaussian", zeroSum = FALSE
    )
    eA <- extCostFunction(x, y, coef(A), alpha, lambda, family = "gaussian")

    A_LS <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = FALSE,
        family = "gaussian", zeroSum = FALSE
    )
    eA_LS <- extCostFunction(x, y, coef(A_LS), alpha, lambda, family = "gaussian")

    eCompA <- extCostFunction(x, y, ref$test_linear$A, alpha, lambda,
        family = "gaussian"
    )

    expect_equal(eA$cost / eCompA$cost, 1.0, tolerance = 1e-2)
    expect_equal(eA$cost / eA_LS$cost, 1.0, tolerance = 1e-2)

    expect_equal(cor(as.numeric(coef(A)), as.numeric(coef(A_LS))),
        1.0,
        tolerance = 1e-5
    )
    expect_equal(cor(as.numeric(coef(A)), ref$test_linear$A),
        1.0,
        tolerance = 1e-5
    )


    ## linear Regression zerosum
    B <- zeroSum(x, y, alpha, lambda,
        algorithm = "CD", standardize = FALSE,
        family = "gaussian"
    )
    eB <- extCostFunction(x, y, coef(B), alpha, lambda, family = "gaussian")

    B_LS <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = FALSE,
        family = "gaussian"
    )
    eB_LS <- extCostFunction(x, y, coef(B_LS), alpha, lambda, family = "gaussian")

    expect_equal(eB$cost / eB_LS$cost, 1.0, 1e-3)
    expect_equal(cor(as.numeric(coef(B)), as.numeric(coef(B_LS))), 1.0,
        tolerance = 1e-4
    )

    expect_equal(sum(coef(B)[-1]), 0.0, tolerance = 1e-13)
    expect_equal(sum(coef(B_LS)[-1]), 0.0, tolerance = 1e-13)


    ## linear Regression standardized
    C <- zeroSum(x, y, alpha, lambda,
        algorithm = "CD", standardize = TRUE,
        family = "gaussian", zeroSum = FALSE
    )
    eC <- extCostFunction(x, y, coef(C), alpha, lambda, family = "gaussian")

    C_LS <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = TRUE,
        family = "gaussian", zeroSum = FALSE
    )
    eC_LS <- extCostFunction(x, y, coef(C_LS), alpha, lambda, family = "gaussian")

    eCompC <- extCostFunction(x, y, ref$test_linear$C, alpha, lambda,
        family = "gaussian"
    )

    ## glmnet packages gives here slightly wore results -> higher tolerance
    expect_equal(eC$cost, eCompC$cost, tolerance = 1e-2)
    expect_equal(cor(as.numeric(coef(C)), ref$test_linear$A), 1.0,
        tolerance = 1e-2
    )

    ## ls seems to perform a little bit better -> higher tolerance
    expect_equal(eC$cost, eC_LS$cost, tolerance = 1e-3)
    expect_equal(cor(as.numeric(coef(C)), as.numeric(coef(C_LS))), 1.0,
        tolerance = 1e-6
    )


    ## linear Regression zerosum standardized
    D <- zeroSum(x, y, alpha, lambda,
        algorithm = "CD", standardize = TRUE,
        family = "gaussian"
    )
    eD <- extCostFunction(x, y, coef(D), alpha, lambda, family = "gaussian")

    D_LS <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = TRUE,
        family = "gaussian"
    )
    eD_LS <- extCostFunction(x, y, coef(D_LS), alpha, lambda, family = "gaussian")

    expect_equal(eD$cost / eD_LS$cost, 1.0, tolerance = 1e-2)
    expect_equal(cor(as.numeric(coef(D)), as.numeric(coef(D_LS))),
        1.0,
        tolerance = 1e-6
    )

    expect_equal(sum(coef(D)[-1]), 0.0, tolerance = 1e-13)
    expect_equal(sum(coef(D_LS)[-1]), 0.0, tolerance = 1e-13)


    ## fusion kernel test
    P <- ncol(x)
    fusion <- Matrix(0, nrow = P - 1, ncol = P, sparse = TRUE)
    for (i in 1:(P - 1)) fusion[i, i] <- 1
    for (i in 1:(P - 1)) fusion[i, (i + 1)] <- -1
    gamma <- 0.30
    lambda <- 0.01
    set.seed(1)

    FA1 <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = FALSE,
        gamma = gamma, fusion = fusion, family = "gaussian",
        zeroSum = FALSE
    )
    eFA1 <- extCostFunction(x, y, coef(FA1), alpha, lambda,
        family = "gaussian",
        gamma = gamma, fusion = fusion
    )

    FA2 <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = FALSE,
        gamma = gamma, fusion = fusion, family = "gaussian"
    )
    eFA2 <- extCostFunction(x, y, coef(FA2), alpha, lambda,
        family = "gaussian",
        gamma = gamma, fusion = fusion
    )

    gamma <- 0.03
    lambda <- 0.001

    FA3 <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = TRUE,
        gamma = gamma, fusion = fusion, family = "gaussian",
        zeroSum = FALSE
    )
    eFA3 <- extCostFunction(x, y, coef(FA3), alpha, lambda,
        family = "gaussian",
        gamma = gamma, fusion = fusion
    )

    FA4 <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = TRUE,
        gamma = gamma, fusion = fusion, family = "gaussian"
    )
    eFA4 <- extCostFunction(x, y, coef(FA4), alpha, lambda,
        family = "gaussian",
        gamma = gamma, fusion = fusion
    )

    costs <- c(eFA1$cost, eFA2$cost, eFA3$cost, eFA4$cost)

    for (i in 1:length(costs))
        expect_lte(costs[i], ref$test_linear$fusionCosts[i] * 1.05)

    # calculate fusion terms and expect that most adjacent features are equal
    fused <- abs(as.numeric(fusion %*% cbind(
        coef(FA1), coef(FA2), coef(FA3),
        coef(FA4)
    )[-1, ]))
    expect_gte(sum(fused < 1e-5) / length(fused), 0.3)
})
