zeroSum
===========
<em>zeroSum</em> is an R package for fitting reference point insensitive linear models. The coefficients are penalized by the elastic net regularization \[1\] and by the zero-sum constraint.

The zero-sum constraint imposes that the sum over the regression weights vanishes. This constraint is 
recommended for log-transformed data where ambiguities in the reference point translate to sample-wise 
shifts. This approach has been proposed in the context of compositional data in \[2\] and in the 
context of reference points in \[4\]. The corresponding minimization problem reads:
<center>
<img src="https://raw.github.com/rehbergT/zeroSum/master/costFunction.png" width="600" />
</center>
The parameter alpha can be used to adjust the ratio between ridge and LASSO regularization. 
For alpha=0 the elastic net becomes
a ridge regularization, for alpha=1  the elastic net becomes
the LASSO regularization.
The function calls of the <em>zeroSum</em> package follow closely 
those of the <em>glmnet</em> package \[3\]. Therefore, the results of the linear models
with or without the zero-sum constraint can be easily compared.

Installation directly from github within R with devtools
--------------------------------------------------------

Open an R session as root and load the [<em>devtools</em>](https://cran.r-project.org/web/packages/devtools/index.html) package:

    library("devtools")
    install_github("rehbergT/zeroSum/zeroSum")

Building and Installation from source
-------------------------------------

The following steps will download the source code of <em>zeroSum</em> install the package:

    # Download the source from github
    git clone https://github.com/rehbergT/zeroSum.git
    cd zeroSum
    # Build the R-package
	R CMD build zeroSum
    # Install the R-package
    R CMD INSTALL zeroSum_0.8.4.tar.gz

### Dependencies

<em>zeroSum</em> requires the following R-packages:
  
  * [<em>foreach</em>](https://cran.r-project.org/web/packages/foreach/index.html)
  * [<em>knitr</em>](https://cran.r-project.org/web/packages/knitr/index.html)
  * [<em>testthat</em>](https://cran.r-project.org/web/packages/testthat/index.html)

Basic Usage
-------------
   
    # Load the zeroSum package
    library(zeroSum)

    # Use the included example data
    X <- log2(exampleData$x+1)
    Y <- exampleData$y
    cv.fit <- zeroSumCVFit( X, Y, alpha=1)
    plot( cv.fit, main="CV-Fit" )

<center>
<img src="https://raw.github.com/rehbergT/zeroSum/master/cvfit.png" width="500"/>
</center>

References
----------
\[1\] H. Zou and T. Hastie. Regularization and variable selection via the elastic
net. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 67(2):301–320, 2005.

\[2\] Wei Lin, Pixu Shi, Rui Feng, and Hongzhe Li. Variable selection in regression with compositional covariates. Biometrika, 2014. doi: 10.1093/biomet/asu031.

\[3\] J. Friedman, T. Hastie, and R. Tibshirani. Regularization paths for gen-
eralized linear models via coordinate descent. Journal of Statistical Soft-
ware, 33(1):1–22, 2010. ISSN 1548-7660. doi: 10.18637/jss.v033.i01.

\[4\] M. Altenbuchinger, T. Rehberg (2016). Reference point insensitive molecular data analysis. 2016 - to be published soon

