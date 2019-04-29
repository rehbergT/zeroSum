context("Testing regression weights")

test_that("regression weights seem to be okay", {
    set.seed(10)

    # an example data set is included in the package
    x <- log2(exampleData$x)
    y <- exampleData$y

    P <- ncol(x)
    N <- nrow(x)

    alpha <- 1.0

    w <- c(
        0.507478203158826, 0.306768506066874, 0.426907666493207, 0.693102080840617,
        0.0851359688676894, 0.225436616456136, 0.274530522990972, 0.272305066231638,
        0.615829307818785, 0.429671525489539, 0.651655666995794, 0.567737752571702,
        0.113508982118219, 0.595925305271521, 0.358049975009635, 0.428809418343008,
        0.0519033221062273, 0.264177667442709, 0.398790730861947, 0.836134143406525,
        0.864721225807443, 0.615352416876704, 0.775109896436334, 0.355568691389635,
        0.405849972041324, 0.706646913895383, 0.838287665275857, 0.239589131204411,
        0.770771533250809, 0.355897744419053, 0.535597037756816
    )
    u <- rep(1, P)
    v <- rep(1, P)

    v[1] <- 0
    u[1] <- 0

    fi <- c(
        3L, 6L, 7L, 2L, 1L, 2L, 4L, 7L, 8L, 7L, 5L, 1L, 4L, 9L, 1L,
        10L, 8L, 2L, 1L, 8L, 3L, 5L, 3L, 9L, 9L, 10L, 10L, 5L, 4L, 6L,
        6L
    )

    A <- zeroSum(x, y,
        alpha = alpha, weights = w, zeroSum.weights = u, foldid = fi,
        penalty.factor = v, threads = 4
    )

    expect_equal(sum(coef(A)[-c(1, 2), ]), 0, tolerance = 1e-12)
    expect_equal(as.numeric(coef(A)[2, 1]), -2.297, tolerance = 1e-2)

    y <- exampleData$ylogistic
    lambda <- 0.01
    u <- rep(1, P)
    v <- rep(1, P)

    v[1] <- 0
    v[19] <- 0
    u[1] <- 0
    u[19] <- 0

    fit1 <- zeroSum(x, y,
        alpha = alpha, lambda = lambda, family = "binomial",
        penalty.factor = v, zeroSum.weights = u
    )
    fit2 <- zeroSum(x, y,
        alpha = alpha, lambda = lambda, family = "binomial",
        algorithm = "LS", penalty.factor = v, zeroSum.weights = u
    )

    expect_equal(sum(coef(fit1)[-c(1, 2, 20), ]), 0, tolerance = 1e-12)
    expect_equal(sum(coef(fit2)[-c(1, 2, 20), ]), 0, tolerance = 1e-12)


    e1 <- extCostFunction(x, y, as.numeric(coef(fit1)), alpha, lambda,
        family = "binomial", penalty.factor = v
    )
    e2 <- extCostFunction(x, y, as.numeric(coef(fit2)), alpha, lambda,
        family = "binomial", penalty.factor = v
    )

    cbind(e1, e2)
    diff <- as.numeric(e1) - as.numeric(e2)
    expect_lte(diff[1], 1e-2)
    expect_lte(diff[3], 1e-2)
    expect_lte(diff[4], 1e-2)
})
