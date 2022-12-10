#' Description of exportToCSV function
#' exports all zeroSum settings for using the c version of zeroSum
#'
#' @import methods
#'
#' @keywords internal
#'
#' @export
exportToCSV <- function(x,
                        y,
                        family = "gaussian",
                        alpha = 1.0,
                        lambda = NULL,
                        lambdaSteps = 100,
                        weights = NULL,
                        penalty.factor = NULL,
                        zeroSum.weights = NULL,
                        nFold = NULL,
                        foldid = NULL,
                        epsilon = NULL,
                        standardize = FALSE,
                        intercept = TRUE,
                        zeroSum = TRUE,
                        threads = "auto",
                        cvStop = 0.1,
                        path = "",
                        name = "",
                        ...) {
    data <- regressionObject(
        x, y, family, alpha, lambda, lambdaSteps, weights,
        penalty.factor, zeroSum.weights, nFold, foldid, epsilon,
        standardize, intercept, zeroSum, threads, cvStop, ...
    )

    utils::write.csv(format(data$x,
        digits = 18, scientific = TRUE,
        trim = TRUE
    ), quote = FALSE, file = paste0(path, "x", name, ".csv"))
    utils::write.csv(format(data$y,
        digits = 18, scientific = TRUE,
        trim = TRUE
    ), quote = FALSE, file = paste0(path, "y", name, ".csv"))

    if (data$type == zeroSumTypes[4, 2]) {
        utils::write.csv(data$status,
            file = paste0(path, "status", name, ".csv")
        )
    }

    if (!is.null(data$fusionC) && data$useFusion) {
        utils::write.csv(data$fusionC,
            file = paste0(path, "fusion", name, ".csv")
        )
    }

    cols <- max(length(data$lambda), length(data$gamma), ncol(x), nrow(x))
    settings <- matrix("", nrow = 29, ncol = cols)
    rownames(settings) <- c(
        "N", "P", "K", "nc", "type", "useZeroSum",
        "useFusion", "useIntercept", "useApprox", "center", "standardize",
        "usePolish", "rotatedUpdates", "precision", "algorithm", "nFold",
        "cvStop", "verbose", "cSum", "alpha", "downScaler", "threads",
        "seed", "w", "u", "v", "lambda", "gamma", "foldid"
    )

    settings[1, 1] <- nrow(data$x)
    settings[2, 1] <- ncol(data$x)
    settings[3, 1] <- ncol(data$y)
    settings[4, 1] <- data$nc
    settings[5, 1] <- data$type
    settings[6, 1] <- data$useZeroSum
    settings[7, 1] <- data$useFusion
    settings[8, 1] <- data$useIntercept
    settings[9, 1] <- data$useApprox
    settings[10, 1] <- data$center
    settings[11, 1] <- data$standardize
    settings[12, 1] <- data$usePolish
    settings[13, 1] <- data$rotatedUpdates
    settings[14, 1] <- format(data$precision,
        digits = 18,
        scientific = TRUE, trim = TRUE
    )
    settings[15, 1] <- data$algorithm
    settings[16, 1] <- data$nFold
    settings[17, 1] <- data$cvStop
    settings[18, 1] <- data$verbose
    settings[19, 1] <- format(data$cSum,
        digits = 18,
        scientific = TRUE, trim = TRUE
    )
    settings[20, 1] <- format(data$alpha,
        digits = 18,
        scientific = TRUE, trim = TRUE
    )
    settings[21, 1] <- format(data$downScaler,
        digits = 18,
        scientific = TRUE, trim = TRUE
    )
    settings[22, 1] <- data$threads
    settings[23, 1] <- data$seed

    settings[24, seq_len(nrow(data$x))] <- format(data$w,
        digits = 18,
        scientific = TRUE, trim = TRUE
    )
    settings[25, seq_len(ncol(data$x))] <- format(data$u,
        digits = 18,
        scientific = TRUE, trim = TRUE
    )
    settings[26, seq_len(ncol(data$x))] <- format(data$v,
        digits = 18,
        scientific = TRUE, trim = TRUE
    )

    settings[27, seq_along(data$lambda)] <- format(data$lambda,
        digits = 18,
        scientific = TRUE, trim = TRUE
    )
    settings[28, seq_along(data$gamma)] <- format(data$gamma,
        digits = 18,
        scientific = TRUE, trim = TRUE
    )
    settings[29, seq_len(nrow(data$x))] <- data$foldid

    utils::write.csv(settings, quote = FALSE, file = paste0(
        path, "settings",
        name, ".csv"
    ), na = "")
    saveRDS(data, file = paste0(path, "rDataObject", name, ".rds"))
}
