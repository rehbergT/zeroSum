library(devtools)
load_all("../zeroSum")
set.seed(3)

X <- log2(exampleData$x)
Y <- exampleData$y
cv.fit <- zeroSum(X, Y, alpha = 1)

png("cvFit.png")
plot(cv.fit)
dev.off()
