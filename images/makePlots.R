    library(zeroSum)
    set.seed(1)

    X <- log2(exampleData$x)
    Y <- exampleData$y
    cv.fit <- zeroSumCVFit( X, Y, alpha=1)

    png("cvFit.png")
    plot( cv.fit, main="CV-Fit" )
    dev.off()

    set.seed(1)
    Y <- exampleData$ylogistic

    png("logisticCvFit.png")
    cv.fit <- zeroSumCVFit( X, Y, type="binomialZS", alpha=1 )
    plot( cv.fit, main="CV-Fit" )
    dev.off()
