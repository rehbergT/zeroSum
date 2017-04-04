# zeroSum


<em>zeroSum</em> is an R package for fitting reference point insensitive linear models by imposing the zero-sum constraint combined with the elastic net
regularization \[3\].


The zero-sum constraint is
recommended for fitting linear models between a response y<sub>i</sub> and log-transformed data x<sub>i</sub> where ambiguities in the reference point translate to sample-wise shifts. The influence of such sample-wise shifts on linear models
is as follows:

<img src="https://raw.github.com/rehbergT/zeroSum/master/images/equation2.png" width="295" />

By restricting the sum of coefficients (red) to zero the model becomes reference point insensitive!

This approach of  has been proposed in the context of compositional data in \[4\] and in the
context of reference points in \[1\]. The corresponding minimization problem reads:

<img src="https://raw.github.com/rehbergT/zeroSum/master/images/equation1.png" width="400" />

where L denotes the log-likelihood function of the regression type.
The parameter &alpha; can be used to adjust the ratio between ridge and LASSO regularization.
For &alpha;=0 the elastic net becomes
a ridge regularization, for &alpha;=1  the elastic net becomes
the LASSO regularization.

For more details about zero-sum see \[1\] and \[2\].

The function calls of the <em>zeroSum</em> package follow closely
those of the <em>glmnet</em> package \[3\]. Therefore, the results of the linear models
with or without the zero-sum constraint can be easily compared.



**Please note that the zero-sum constraint only yields reference point insensitive models on log transformed data!**

## Table of contents
  * [Installation](#installation)
    * [Using devtools](#using-devtools)
    * [Manual installation](#manual-installation)
    * [Building from source](#building-from-source)
    * [Performance Advice](#performance-advice)
  * [Dependencies](#dependencies)
  * [Quick start](#quick-start)
  * [Supported Regression Types](#supported-regression-types)
  * [Standalone HPC version](#standalone-hpc-version)
      * [Build Instructions](#build-instructions)
      * [Basic usage](#basic-usage)
  * [References](#references)


## Installation

### Using devtools
##### Windows
Open an R session as admin and load the [<em>devtools</em>](https://cran.r-project.org/web/packages/devtools/index.html) package:

    library("devtools")
    install_github("rehbergT/zeroSum/zeroSum_1.0.0.zip")

##### Linux / OS X
Open an R session as root and load the [<em>devtools</em>](https://cran.r-project.org/web/packages/devtools/index.html) package:

    library("devtools")
    install_github("rehbergT/zeroSum/zeroSum")


### Manual Installation
##### Windows
Download the precompiled [<em>windows package</em>](https://github.com/rehbergT/zeroSum/zeroSum_1.0.0.zip) (a zip file), open an
R session as admin, navigate to the folder with the zip file and install it with:

    install.packages("zeroSum_1.0.0.zip", repos = NULL)

##### Linux
Download the [<em>package</em>](https://github.com/rehbergT/zeroSum/zeroSum_1.0.0.tar.gz), open an
R session as admin, navigate to the folder with the tar.gz file and install it with:

    install.packages("zeroSum_1.0.0.tar.gz", repos = NULL)



### Building from source
Open a terminal and download the source from github (git client needs to be installed, or downloaded manually):

    git clone https://github.com/rehbergT/zeroSum.git

Change the working directory to the downloaded folder:

    cd zeroSum

Build and install the created package:

    R CMD build zeroSum
    R CMD INSTALL zeroSum_1.0.0.tar.gz

### Performance advice

**The following is only supported on Linux / OS X**


The crucial parts of <em>zeroSum</em> can make use of AVX(2)/AVX512 instructions.
If your CPU supports these instructions significant performance boosts can be
 achieved.

<em>zeroSum</em> can recognize at compile-time which instructions are
available and uses them automatically. However, your compiler settings need be changed to build for a specific CPU architecture. This can be done by creating a file Makevars in the folder .R  in your home directory:

**This command overwrites the ~/.R/Makevars file!**

    echo "CXX1XFLAGS += -march=native -mtune=native -O3 -flto" > ~/.R/Makevars

<em>native</em> means that code will be compiled for your current CPU architecture.
This command also enables further compiler optimizations (O3, mtune=native) and link time optimizations (flto).

The following command(s) can be used to check which AVX instructions are supported by your compiler and CPU.
For clang:

    clang -march=native -dM -E - < /dev/null | egrep "AVX"

For gcc:

    gcc -march=native -dM -E - < /dev/null | egrep "AVX"

Example:

    $ gcc -march=native -dM -E - < /dev/null | egrep "AVX"
    #define __AVX__ 1
    #define __AVX2__ 1

this means AVX and AVX2 is supported by the CPU.

## Dependencies

  *  <em>zeroSum</em> requires a modern compiler which supports C++11 and openMP
  * Moreover R-package [<em>testthat</em>](https://cran.r-project.org/web/packages/testthat/index.html) is needed

## Quick start

Load the <em>zeroSum</em> package and the included example data

    library(zeroSum)
    set.seed(1)
    X <- log2(exampleData$x)
    Y <- exampleData$y

Perform a cross-validation regression for an automatically approximated &lambda;
sequence with the LASSO (&alpha;=1):

    cv.fit <- zeroSumCVFit( X, Y, alpha=1)


use the plot() function to see the CV-error versus regularization strength &lambda;

    plot( cv.fit, main="CV-Fit" )

<center>
<img src="https://raw.github.com/rehbergT/zeroSum/master/images/cvFit.png" width="500"/>
</center>

The coef() function can be used to extract the coefficients for a specific &lambda;

    head( coef(cv.fit, s="lambda.min") )
        [,1]
    intercept 27.3026105
    Feature1   0.0000000
    Feature2   0.7558415
    Feature3   0.0000000
    Feature4   0.3207759
    Feature5   0.5220437


Note that the sum of coefficients should be zero:

    sum( coef(cv.fit)[-1,] ) ## -1 to remove the intercept
    8.326673e-17

Use the predict() function to obtain new predictions with the cv.fit:  

    head( predict( cv.fit, newx=X) )
        [,1]
    Sample1 -3.435823
    Sample2 46.935922
    Sample3 32.651803
    Sample4 35.503812
    Sample5 44.249014
    Sample6 61.353946



## Supported regression Types

### Overview

<em>zeroSum</em> supports linear regression (gaussian response) and logistic regression (binomial response).
<!-- <em>zeroSum</em> supports linear regression (gaussian response), logistic regression (binomial response) and multinomial regression. -->

The functions zeroSumFit() and zeroSumCVFit() have the parameter <em>type</em> with which the regression type can be determined.
Models for these types can be created with and without the zero-sum constraint. Types with the zero-sum constraint are denoted with
the suffix ZS.
<!-- Moreover fused and fusion regression are available and can be enabled with the prefix <em>fused</em> or <em>fusion</em>  -->

<!-- **Please note that only the types gaussian, gaussianZS, binomial and binomialZS are well tested and all other are experimental!** -->

 The following different types are supported:

| **normal regression** | **zero-sum**     |
| --------------------- | ---------------- |
| gaussian              |  gaussianZS      |
| binomial              |  binomialZS      |

### Example: zero-sum logistic regression

First, the <em>zeroSum</em> package and the included binomial example data is loaded. Second, a cross-validation regression is performed. The only difference
is that the type of the regression has to be specified with the parameter <em>type</em>:

    library(zeroSum)
    set.seed(1)
    X <- log2(exampleData$x)
    Y <- exampleData$ylogistic
    cv.fit <- zeroSumCVFit( X, Y, type="binomialZS", alpha=1 )

The sum of the coefficients should again be zero:

    sum( coef(cv.fit)[-1,] )
    -1.542676e-16

The predict function now returns the probability of the sample i being class 1 Pr(G=1|x<sub>i</sub>)

    head( predict( cv.fit, newx=X) )
        [,1]
    [1,] 0.95936135
    [2,] 0.09293259
    [3,] 0.44076669
    [4,] 0.30686124
    [5,] 0.04310259
    [6,] 0.03909084



## Standalone HPC (high performance computing) version

<em>zeroSum</em> is available as a standalone C++ program which allows
an easy utilization on computing clusters.

### Build instructions
Open a terminal and download the source from github (git client needs to be installed, or downloaded manually):

    git clone https://github.com/rehbergT/zeroSum.git

Change the working directory to the downloaded folder:

    cd zeroSum/hpc_version

Build the <em>zeroSum</em> hpc version with:

    make

As explained in [performance advice](#performance-advice) AVX instructions can be used when your compiler and your CPU architecture
supports them. By default the Makefile is configured  to compile <em>zeroSum</em> for the currently used architecture (<em>native</em>).
This can be adjusted in the beginning of the Makefile.

### Basic usage

We provide R functions for exporting all necessary files for the standalone HPC version and for importing the generated results.


Open a terminal and create an empty folder. Change the working directory to the created folder. As shown above, load <em>zeroSum</em> and the example data:

    library(zeroSum)
    set.seed(1)
    x <- log2(exampleData$x)
    y <- exampleData$y

The exportRegressionDataToCSV() function has the same arguments as zeroSumCVFit() but with the additional arguments <em>name</em> and <em>path</em>.
With <em>name</em> one can set the prefix name of all exported files and with <em>path</em> the path for exporting the files can be determined.

    exportRegressionDataToCSV( x, y, name="test", path="./" )

4 files are now generated:

    $ ls
    rDataObject_test.rds  settingstest.csv  xtest.csv  ytest.csv

The .rds file is only necessary for importing the results and only the other 3 have to be passed to the HPC <em>zeroSum</em> version as follows:

    mpirun -np 1 /path/to/hpcVersion/zeroSum settingstest.csv xtest.csv ytest.csv ./ example

The last two arguments determine the path where the results should be saved and the of name the file in which the results should be saved:

    $ ls
    example_0_stats.csv  rDataObject_test.rds  settingstest.csv  xtest.csv  ytest.csv

As one can see the file example_0_stats.csv containing the results has been created.
One can now import the file back into R with the importRegressionDataFromCSV() function:

    library(zeroSum)
    cv.fit <- importRegressionDataFromCSV("rDataObject_test.rds", "example" )

    head( coef(cv.fit, s="lambda.min") )
                    [,1]
    intercept 27.3026110
    Feature1   0.0000000
    Feature2   0.7558415
    Feature3   0.0000000
    Feature4   0.3207761
    Feature5   0.5220436

The same coefficients as above have been obtained (despite some numerical uncertainty).


## References
\[1\] M. Altenbuchinger, T. Rehberg, H. U. Zacharias, F. Staemmler, K. Dettmer, D. Weber, A. Hiergeist, A. Gessner, E. Holler, P. J. Oefner, R. Spang. Reference point insensitive molecular data analysis. Bioinformatics 33(2):219, 2017. doi: 10.1093/bioinformatics/btw598

\[2\] H. U. Zacharias, T. Rehberg, S. Mehrl, D. Riechtman, T. Wettig, P. J. Oefner, R. Spang, W. Gronwald, M. Altenbuchinger. Scale-invariant biomarker discovery in urine and plasma metabolite fingerprints. ArXiv e-prints. <http://arxiv.org/abs/1703.07724>

\[3\] H. Zou and T. Hastie. Regularization and variable selection via the elastic
net. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 67(2):301–320, 2005.

\[4\] Wei Lin, Pixu Shi, Rui Feng, and Hongzhe Li. Variable selection in regression with compositional covariates. Biometrika, 2014. doi: 10.1093/biomet/asu031.

\[5\] J. Friedman, T. Hastie, and R. Tibshirani. Regularization paths for generalized linear models via coordinate descent. Journal of Statistical Software, 33(1):1–22, 2010. ISSN 1548-7660. doi: 10.18637/jss.v033.i01.
