context("Testing cox regression")

test_that("cox regression test", {
    ref <- readRDS("references.rds")
    x <- log2(exampleData$x)
    y <- exampleData$yCox
    set.seed(4)
    w <- runif(nrow(x))
    alpha <- 0.5
    lambda <- 0.2

    A <- zeroSum(x, y, alpha, lambda,
        family = "cox", standardize = FALSE,
        weights = w, zeroSum = FALSE
    )
    eA <- extCostFunction(x, y, coef(A), alpha, lambda, family = "cox")

    A_LS <- zeroSum(x, y, alpha, lambda,
        family = "cox", standardize = FALSE,
        algorithm = "LS", weights = w, zeroSum = FALSE
    )
    eA_LS <- extCostFunction(x, y, coef(A_LS), alpha, lambda, family = "cox")

    A_LS2 <- zeroSum(x, y, alpha, lambda,
        family = "cox", standardize = FALSE,
        algorithm = "LS", useApprox = FALSE, weights = w,
        zeroSum = FALSE
    )
    eA_LS2 <- extCostFunction(x, y, coef(A_LS2), alpha, lambda, family = "cox")

    eCompA <- extCostFunction(x, y, ref$test_cox$A, alpha, lambda,
        family = "cox"
    )

    expect_equal(eA$cost, eA_LS$cost, tolerance = 1e-4)
    expect_equal(cor(as.numeric(coef(A)), as.numeric(coef(A_LS))),
        1.0,
        tolerance = 1e-3
    )

    expect_equal(eA$cost, eA_LS2$cost, tolerance = 1e-4)
    expect_equal(cor(as.numeric(coef(A)), as.numeric(coef(A_LS2))), 1.0,
        tolerance = 1e-2
    )

    expect_equal(eA$cost, eCompA$cost, tolerance = 1e-4)
    expect_equal(cor(as.numeric(coef(A)), as.numeric(ref$test_cox$A)), 1.0,
        tolerance = 1e-3
    )


    B <- zeroSum(x, y, alpha, lambda,
        standardize = FALSE,
        family = "cox"
    )
    eB <- extCostFunction(x, y, coef(B), alpha, lambda, family = "cox")

    B_LS <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = FALSE,
        family = "cox"
    )
    eB_LS <- extCostFunction(x, y, coef(B_LS), alpha, lambda, family = "cox")

    B_LS2 <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = FALSE,
        useApprox = FALSE, family = "cox"
    )
    eB_LS2 <- extCostFunction(x, y, coef(B_LS2), alpha, lambda, family = "cox")


    expect_equal(eB$cost, eB_LS$cost, tolerance = 0.05)
    expect_equal(eB$cost, eB_LS2$cost, tolerance = 0.05)

    expect_equal(cor(as.numeric(coef(B)), as.numeric(coef(B_LS))), 1.0,
        tolerance = 1e-3
    )
    expect_equal(cor(as.numeric(coef(B_LS)), as.numeric(coef(B_LS2))), 1.0,
        tolerance = 1e-3
    )

    expect_equal(sum(coef(B)[-1]), 0.0, tolerance = 1e-14)
    expect_equal(sum(coef(B_LS)[-1]), 0.0, tolerance = 1e-14)



    C <- zeroSum(x, y, alpha, lambda,
        standardize = TRUE,
        family = "cox", zeroSum = FALSE
    )
    eC <- extCostFunction(x, y, coef(C), alpha, lambda, family = "cox")

    C_LS <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = TRUE,
        family = "cox", zeroSum = FALSE
    )
    eC_LS <- extCostFunction(x, y, coef(C_LS), alpha, lambda, family = "cox")

    C_LS2 <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = TRUE,
        useApprox = FALSE, family = "cox", zeroSum = FALSE
    )
    eC_LS2 <- extCostFunction(x, y, coef(C_LS2), alpha, lambda, family = "cox")

    eCompC <- extCostFunction(x, y, ref$test_cox$C, alpha,
        lambda,
        family = "cox"
    )

    ## glmnet packages gives here slightly wore results -> higher tolerance
    expect_equal(eC$cost, eCompC$cost, tolerance = 1e-3)
    expect_equal(cor(as.numeric(coef(C)), as.numeric(ref$test_cox$C)), 1.0,
        tolerance = 1e-3
    )

    ## ls seems to perform a little bit better -> higher tolerance
    expect_equal(eC$cost, eC_LS$cost, tolerance = 1e-3)
    expect_equal(cor(as.numeric(coef(C)), as.numeric(coef(C_LS))), 1.0,
        tolerance = 1e-3
    )

    expect_equal(eC$cost, eC_LS2$cost, tolerance = 1e-3)
    expect_equal(cor(as.numeric(coef(C)), as.numeric(coef(C_LS2))), 1.0,
        tolerance = 1e-2
    )


    D <- zeroSum(x, y, alpha, lambda,
        standardize = TRUE,
        family = "cox"
    )
    eD <- extCostFunction(x, y, coef(D), alpha, lambda, family = "cox")

    D_LS <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = TRUE,
        family = "cox"
    )
    eD_LS <- extCostFunction(x, y, coef(D_LS), alpha, lambda, family = "cox")

    D_LS2 <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = TRUE,
        useApprox = FALSE, family = "cox"
    )
    eD_LS2 <- extCostFunction(x, y, coef(D_LS2), alpha, lambda, family = "cox")

    expect_equal(eD$cost / eD_LS$cost, 1.0, tolerance = 1e-3)
    expect_equal(cor(as.numeric(coef(D)), as.numeric(coef(D_LS))), 1.0,
        tolerance = 1e-3
    )

    expect_equal(eD$cost / eD_LS2$cost, 1.0, tolerance = 1e-3)
    expect_equal(cor(as.numeric(coef(D)), as.numeric(coef(D_LS2))), 1.0,
        tolerance = 1e-4
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
        algorithm = "LS", gamma = gamma,
        fusion = fusion, family = "cox", zeroSum = FALSE
    )
    eFA1 <- extCostFunction(x, y, coef(FA1), alpha, lambda,
        family = "cox",
        gamma = gamma, fusion = fusion
    )

    FA2 <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", gamma = gamma,
        fusion = fusion, useApprox = FALSE,
        family = "cox", zeroSum = FALSE
    )
    eFA2 <- extCostFunction(x, y, coef(FA2), alpha, lambda,
        family = "cox",
        gamma = gamma, fusion = fusion
    )

    FA3 <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", gamma = gamma,
        fusion = fusion, family = "cox"
    )
    eFA3 <- extCostFunction(x, y, coef(FA3), alpha, lambda,
        family = "cox",
        gamma = gamma, fusion = fusion
    )

    FA4 <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", gamma = gamma,
        fusion = fusion, useApprox = FALSE, family = "cox"
    )
    eFA4 <- extCostFunction(x, y, coef(FA4), alpha, lambda,
        family = "cox",
        gamma = gamma, fusion = fusion
    )

    FA5 <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = TRUE,
        gamma = gamma, fusion = fusion, family = "cox", zeroSum = FALSE
    )
    eFA5 <- extCostFunction(x, y, coef(FA5), alpha, lambda,
        family = "cox",
        gamma = gamma, fusion = fusion
    )

    FA6 <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = TRUE,
        gamma = gamma, fusion = fusion, useApprox = FALSE,
        family = "cox", zeroSum = FALSE
    )
    eFA6 <- extCostFunction(x, y, coef(FA6), alpha, lambda,
        family = "cox",
        gamma = gamma, fusion = fusion
    )

    FA7 <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = TRUE,
        gamma = gamma, fusion = fusion, family = "cox"
    )
    eFA7 <- extCostFunction(x, y, coef(FA7), alpha, lambda,
        family = "cox",
        gamma = gamma, fusion = fusion
    )

    FA8 <- zeroSum(x, y, alpha, lambda,
        algorithm = "LS", standardize = TRUE,
        gamma = gamma, fusion = fusion, useApprox = FALSE, family = "cox"
    )
    eFA8 <- extCostFunction(x, y, coef(FA8), alpha, lambda,
        family = "cox",
        gamma = gamma, fusion = fusion
    )


    costs <- c(
        eFA1$cost, eFA2$cost, eFA3$cost, eFA4$cost, eFA5$cost, eFA6$cost,
        eFA7$cost, eFA8$cost
    )

    for (i in seq_len(length(costs))) {
        expect_lte(costs[i], ref$test_cox$fusionCosts[i] + 5e-2)
    }

    # calculate fusion terms and expect that adjacent features are equal
    fused <- abs(as.numeric(fusion %*% cbind(
        coef(FA1), coef(FA2), coef(FA3),
        coef(FA4), coef(FA5), coef(FA6), coef(FA7), coef(FA8)
    )[-1, ]))
    expect_gte(sum(fused < 1e-3) / length(fused), 0.3)
})
