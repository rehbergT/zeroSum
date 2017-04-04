#' Description of importRegressionDataFromCSV function
#' imports zeroSum cvfit csv from the c version of zeroSum
#'
#' @keywords internal
#'
#' @export
importRegressionDataFromCSV <- function( rDataObj, cOutput )
{
    data <- readRDS(rDataObj)

    content <- NULL
    i <- 0
    while(TRUE)
    {
        print(i)
        file <- paste0(cOutput, "_", i, "_stats.csv" )

        if( file.exists(file) ){
            fcont <- utils::read.csv( file, header=FALSE)
            content <- rbind( content, fcont )
            i <- i+1
        } else
        {
            break
        }
    }

    content <- content[,-ncol(content)] ## last col is na, its just a ","

    t2 <- t(content)

    data$result <- as.numeric(t2)

    fit <- zeroSumCVFitObject(data)

    return(fit)
}
