context("Testing multinomial regression")

test_that("multinomial regression equals glmnet", {
    ref <- readRDS("references.rds")
    x <- log2(exampleData$x)
    y <- exampleData$yMultinomial
    P <- ncol(x)
    K <- max(y)
    alpha <- 0.5
    lambda <- 0.01
    set.seed(1)

    ## multinomial Regression
    A <- zeroSum(x, y, alpha, lambda, standardize = FALSE,
                 family = "multinomial", zeroSum = FALSE)
    eA <- extCostFunction(x, y, coef(A), alpha, lambda, family = "multinomial")

    A_LS <- zeroSum(x, y, alpha, lambda, algorithm = "LS", standardize = FALSE,
                    useApprox = TRUE, family = "multinomial",
                    zeroSum = FALSE)
    eA_LS <- extCostFunction(x, y, coef(A_LS), alpha, lambda,
                             family = "multinomial")

    A_LS2 <- zeroSum(x, y, alpha, lambda, algorithm = "LS", standardize = FALSE,
                     useApprox = FALSE, family = "multinomial",
                     zeroSum = FALSE)
    eA_LS2 <- extCostFunction(x, y, coef(A_LS2), alpha, lambda,
                              family = "multinomial")

    eCompA <- extCostFunction(x, y, ref$test_multi$A, alpha, lambda,
                              family = "multinomial")


    expect_equal(eA$cost, eA_LS$cost, tolerance = 1e-3)
    expect_equal(cor(as.numeric(coef(A)), as.numeric(coef(A_LS))), 1.0,
                 tolerance = 1e-3)

    expect_equal(eA$cost, eA_LS2$cost, tolerance = 1e-3)
    expect_equal(cor(as.numeric(coef(A)), as.numeric(coef(A_LS2))), 1.0,
                 tolerance = 1e-3)

    expect_equal(eA$cost, eCompA$cost, tolerance = 1e-3)
    expect_equal(cor(as.numeric(coef(A)[-1, ]),
                 as.numeric(ref$test_multi$A[-1, ])), 1.0, tolerance = 0.05)


    # ## multinomial Regression zerosum
    B <- zeroSum(x, y, alpha, lambda, standardize = FALSE,
                 family = "multinomial", precision = 1e-10)
    eB <- extCostFunction(x, y, coef(B), alpha, lambda, family = "multinomial")

    B_LS <- zeroSum(x, y, alpha, lambda, algorithm = "LS", standardize = FALSE,
                    family = "multinomial", precision = 1e-10)
    eB_LS <- extCostFunction(x, y, coef(B_LS), alpha, lambda,
                             family = "multinomial")

    B_LS2 <- zeroSum(x, y, alpha, lambda, algorithm = "LS", standardize = FALSE,
                     useApprox = FALSE, family = "multinomial",
                     precision = 1e-10)
    eB_LS2 <- extCostFunction(x, y, coef(B_LS2), alpha, lambda,
                              family = "multinomial")

    expect_lte(eB$cost,     0.334)
    expect_lte(eB_LS$cost,  0.336)
    expect_lte(eB_LS2$cost, 0.337)

    expect_equal(cor(as.numeric(coef(B)), as.numeric(coef(B_LS))), 0.99,
                 tolerance = 0.05)
    expect_equal(cor(as.numeric(coef(B_LS)), as.numeric(coef(B_LS2))), 1.0,
                 tolerance = 0.05)

    expect_equal(sum(coef(B)[-1, 1]), sum(coef(B)[-1, 2]), tolerance = 1e-12)
    expect_equal(sum(coef(B)[-1, 1]), sum(coef(B)[-1, 3]), tolerance = 1e-12)

    expect_equal(sum(coef(B_LS)[-1, 1]), sum(coef(B_LS)[-1, 2]),
                 tolerance = 1e-12)
    expect_equal(sum(coef(B_LS)[-1, 1]), sum(coef(B_LS)[-1, 3]),
                 tolerance = 1e-12)

    expect_equal(sum(coef(B_LS2)[-1, 1]), sum(coef(B_LS2)[-1, 2]),
                 tolerance = 1e-12)
    expect_equal(sum(coef(B_LS2)[-1, 1]), sum(coef(B_LS2)[-1, 3]),
                 tolerance = 1e-12)

    ## multinomial Regression standardized
    C <- zeroSum(x, y, alpha, lambda, standardize = TRUE,
                 family = "multinomial", zeroSum = FALSE)
    eC <- extCostFunction(x, y, coef(C), alpha, lambda, family = "multinomial")

    C_LS <- zeroSum(x, y, alpha, lambda, algorithm = "LS", standardize = TRUE,
                    family = "multinomial", zeroSum = FALSE)
    eC_LS <- extCostFunction(x, y, coef(C_LS), alpha, lambda,
                             family = "multinomial")

    C_LS2 <- zeroSum(x, y, alpha, lambda, algorithm = "LS", standardize = TRUE,
                     useApprox = FALSE, family = "multinomial", zeroSum = FALSE)
    eC_LS2 <- extCostFunction(x, y, coef(C_LS2), alpha, lambda,
                              family = "multinomial")


    eCompC <- extCostFunction(x, y, ref$test_multi$C, alpha, lambda,
                              family = "multinomial")

    ## glmnet packages gives here slightly wore results -> higher tolerance
    expect_equal(eC$cost, eCompC$cost, tolerance = 1e-3)
    expect_equal(cor(as.numeric(coef(C)[-1, ]),
                 as.numeric(ref$test_multi$C[-1, ])), 1.0, tolerance = 1e-2)

    ## ls seems to perform a little bit better -> higher tolerance
    expect_equal(eC$cost, eC_LS$cost, tolerance = 1e-3)
    expect_equal(cor(as.numeric(coef(C)[-1, ]), as.numeric(coef(C_LS)[-1, ])),
                 1.0, tolerance = 1e-2)

    expect_equal(eC$cost, eC_LS2$cost, tolerance = 1e-3)
    expect_equal(cor(as.numeric(coef(C)[-1, ]), as.numeric(coef(C_LS2)[-1, ])),
                 1.0, tolerance = 1e-2)

    ## multinomial Regression zerosum standardized
    D <- zeroSum(x, y, alpha, lambda, standardize = TRUE,
                 family = "multinomial")
    eD <- extCostFunction(x, y, coef(D), alpha, lambda, family = "multinomial")

    D_LS <- zeroSum(x, y, alpha, lambda, algorithm = "LS", standardize = TRUE,
                    family = "multinomial")
    eD_LS <- extCostFunction(x, y, coef(D_LS), alpha, lambda,
                             family = "multinomial")

    D_LS2 <- zeroSum(x, y, alpha, lambda, algorithm = "LS", standardize = TRUE,
                     useApprox = FALSE, family= "multinomial")
    eD_LS2 <- extCostFunction(x, y, coef(D_LS2), alpha, lambda,
                              family = "multinomial")


    expect_equal(eD$cost / eD_LS$cost, 1.0, tolerance = 5e-2)
    expect_equal(cor(as.numeric(coef(D)[-1, ]), as.numeric(coef(D_LS)[-1, ])),
                 1.0, tolerance = 5e-2)

    expect_equal(eD$cost / eD_LS2$cost, 1.0, tolerance = 5e-2)
    expect_equal(cor(as.numeric(coef(D)[-1, ]), as.numeric(coef(D_LS2)[-1, ])),
                 1.0, tolerance = 5e-2)

    expect_equal(sum(coef(D)[-1, 1]), sum(coef(D)[-1, 2]), tolerance = 1e-12)
    expect_equal(sum(coef(D)[-1, 1]), sum(coef(D)[-1, 3]), tolerance = 1e-12)

    expect_equal(sum(coef(D_LS)[-1, 1]), sum(coef(D_LS)[-1, 2]),
                 tolerance = 1e-12)
    expect_equal(sum(coef(D_LS)[-1, 1]), sum(coef(D_LS)[-1, 3]),
                 tolerance = 1e-12)

    expect_equal(sum(coef(D_LS2)[-1, 1]), sum(coef(D_LS2)[-1, 2]),
                 tolerance = 1e-12)
    expect_equal(sum(coef(D_LS2)[-1, 1]), sum(coef(D_LS2)[-1, 3]),
                 tolerance = 1e-12)


    ## fusion kernel test
    fusion <- Matrix(0, nrow = P - 1, ncol = P, sparse = TRUE)
    for (i in 1:(P - 1)) fusion[i, i] <-  1
    for (i in 1:(P - 1)) fusion[i, (i + 1)] <- -1
    gamma <- 0.03
    lambda <- 0.001
    set.seed(1)

    FA1 <- zeroSum(x, y, alpha, lambda, algorithm = "LS", gamma = gamma, fusion = fusion, family = "multinomial", zeroSum = FALSE)
    eFA1 <- extCostFunction(x, y, coef(FA1), alpha, lambda, family = "multinomial", gamma = gamma, fusion = fusion)

    FA2 <- zeroSum(x, y, alpha, lambda, algorithm = "LS", gamma = gamma, fusion = fusion, useApprox = FALSE, family = "multinomial", zeroSum = FALSE)
    eFA2 <- extCostFunction(x, y, coef(FA2), alpha, lambda, family = "multinomial", gamma = gamma, fusion = fusion)

    FA3 <- zeroSum(x, y, alpha, lambda, algorithm = "LS", gamma = gamma, fusion = fusion, family = "multinomial")
    eFA3 <- extCostFunction(x, y, coef(FA3), alpha, lambda, family = "multinomial", gamma = gamma, fusion = fusion)

    FA4 <- zeroSum(x, y, alpha, lambda, algorithm = "LS", gamma = gamma, fusion = fusion, useApprox = FALSE, family = "multinomial")
    eFA4 <- extCostFunction(x, y, coef(FA4), alpha, lambda, family = "multinomial", gamma = gamma, fusion = fusion)

    FA5 <- zeroSum(x, y, alpha, lambda, algorithm = "LS", standardize = TRUE, gamma = gamma, fusion = fusion, family = "multinomial", zeroSum = FALSE)
    eFA5 <- extCostFunction(x, y, coef(FA5), alpha, lambda, family = "multinomial", gamma = gamma, fusion = fusion)

    FA6 <- zeroSum(x, y, alpha, lambda, algorithm = "LS", standardize = TRUE, gamma = gamma, fusion = fusion, useApprox = FALSE, family = "multinomial", zeroSum = FALSE)
    eFA6 <- extCostFunction(x, y, coef(FA6), alpha, lambda, family = "multinomial", gamma = gamma, fusion = fusion)

    FA7 <- zeroSum(x, y, alpha, lambda, algorithm = "LS", standardize = TRUE, gamma = gamma, fusion = fusion, family = "multinomial")
    eFA7 <- extCostFunction(x, y, coef(FA7), alpha, lambda, family = "multinomial", gamma = gamma, fusion = fusion)

    FA8 <- zeroSum(x, y, alpha, lambda, algorithm = "LS", standardize = TRUE, gamma = gamma, fusion = fusion, useApprox = FALSE, family = "multinomial")
    eFA8 <- extCostFunction(x, y, coef(FA8), alpha, lambda, family = "multinomial", gamma = gamma, fusion = fusion)

    costs <- c(eFA1$cost, eFA2$cost, eFA3$cost, eFA4$cost, eFA5$cost, eFA6$cost, eFA7$cost, eFA8$cost)

    for (i in 1:length(costs)) {
        expect_lte(costs[i], ref$test_multi$fusionCosts[i] + 5e-2)
    }
    # calculate fusion terms and expect that most are adjacent features are equal
    fused <- abs(as.numeric(fusion %*% cbind(coef(FA1), coef(FA2), coef(FA3), coef(FA4), coef(FA5), coef(FA6), coef(FA7), coef(FA8))[-1, ]))
    expect_gte(sum(fused < 1e-5) / length(fused), 0.5)

})
