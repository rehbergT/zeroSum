

    context("Testing zeroSum regression")

    test_that( "zeroSum regression seems to work",{

        x <- log2(exampleData$x+1)
        y <- exampleData$y
        alpha <- 1    

        nFold <- 10
        N <- nrow(x)
        set.seed(1)
        foldid=foldid <- sample(rep( rep(1:nFold), length.out=N))

        set.seed(1)
        elnet.fit <- zeroSumCVFit( x, y, alpha=alpha, cvStop=FALSE,
                                    type="elNet",foldid=foldid)
    
        set.seed(2)
        zeroSum.fit <- zeroSumCVFit( x, y, alpha=alpha, cvStop=FALSE,
                                      type="zeroSumElNet",foldid=foldid)
    

        zeroSum.comp <- readRDS( "zeroSumFit_verified.rda" )
        elNet.comp   <- readRDS( "elnetFit_verified.rda" )     

        expect_that( elnet.fit,   equals(elNet.comp, tolerance=1e-3) )
        expect_that( zeroSum.fit, equals(zeroSum.comp, tolerance=1e-3) )


    } )
