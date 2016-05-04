

    context("Testing zeroSum regression")

    test_that( "zeroSum regression seems to work",{


        data <- readRDS( "data.rda" )

        x <- data$x
        y <- data$y
        alpha <- 1  
        foldid <- c(4L, 9L, 10L, 4L, 9L, 7L, 4L, 10L, 7L, 1L, 2L, 5L, 3L, 8L, 3L, 
                1L, 7L, 9L, 1L, 8L, 9L, 2L, 6L, 3L, 3L, 8L, 2L, 10L, 10L, 5L, 
                7L, 5L, 1L, 5L, 6L, 2L, 6L, 6L, 8L, 4L)

        

        set.seed(1)
        elnet.fit <- zeroSumCVFit( x, y, alpha=alpha, devianceStop=FALSE,
                                    type="elNet",foldid=foldid)
    
        set.seed(2)
        zeroSum.fit <- zeroSumCVFit( x, y, alpha=alpha, devianceStop=FALSE,
                                      type="zeroSumElNet",foldid=foldid)
    

        zeroSum.comp <- readRDS( "zeroSumFit_verified.rda" )
        elNet.comp   <- readRDS( "elnetFit_verified.rda" )     

        expect_that( elnet.fit,   equals(elNet.comp, tolerance=1e-3) )
        expect_that( zeroSum.fit, equals(zeroSum.comp, tolerance=1e-3) )


    } )
