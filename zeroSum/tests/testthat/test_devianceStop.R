
    context("Testing deviance stop")

    test_that( "deviance stop works",  {

        declining <- function( numUp, numDown, numUp2, numDown2 )
        {
            ratio <- c( seq(0.1, 0.9, length.out=numUp ),
                        seq(0.9, 0.1, length.out=numDown ),
                        seq(0.1, 0.9, length.out=numUp2 ),
                        seq(0.9, 0.1, length.out=numDown2 ) )
            return(ratio)
        }


        devRatio1 <- declining(5,5,0,0)
    
        expect_true(  devianceStop( devRatio1, 5) )    
        expect_false( devianceStop( devRatio1, 6) )


        devRatio2 <- declining(5,4,2,0)

        expect_false( devianceStop( devRatio2, 1) )    
        expect_false( devianceStop( devRatio2, 5) )


        devRatio3 <- declining(3,2,2,7)

        expect_true(  devianceStop( devRatio3, 3) )    
        expect_false( devianceStop( devRatio3, 8) )


        devRatio4 <- declining(10,50,5,10)

        expect_true(  devianceStop( devRatio4, 10) )    
        expect_false( devianceStop( devRatio4, 11) )


        devRatio5 <- declining(0,5,2,9)

        expect_true(  devianceStop( devRatio5, 3) )    
        expect_false( devianceStop( devRatio5, 10) )


    })
     
