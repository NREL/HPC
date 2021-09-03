$Title  A Transportation Problem (TRNSPORT,SEQ=1)
$Ontext

This problem finds a least cost shipping schedule that meets
requirements at markets and supplies at factories.


Dantzig, G B, Chapter 3.3. In Linear Programming and Extensions.
Princeton University Press, Princeton, New Jersey, 1963.

This formulation is described in detail in:
Rosenthal, R E, Chapter 2: A GAMS Tutorial. In GAMS: A User's Guide.
The Scientific Press, Redwood City, California, 1988.

The line numbers will not match those in the book because of these
comments.

$Offtext


  Sets
       i   canning plants   / seattle, san-diego /
       j   markets          / new-york, chicago, topeka / ;

  Parameters

       a(i)  capacity of plant i in cases
         /    seattle     350
              san-diego   600  /

       b(j)  demand at market j in cases
         /    new-york    325
              chicago     300
              topeka      275  / ;

  Table d(i,j)  distance in thousands of miles
                    new-york       chicago      topeka
      seattle          2.5           1.7          1.8
      san-diego        2.5           1.8          1.4  ;

  Scalar f  freight in dollars per case per thousand miles  /90/ ;

  Parameter c(i,j)  transport cost in thousands of dollars per case ;

            c(i,j) = f * d(i,j) / 1000 ;

  Variables
       x(i,j)  shipment quantities in cases
       z       total transportation costs in thousands of dollars ;

  Positive Variable x ;

  Equations
       cost        define objective function
       supply(i)   observe supply limit at plant i
       demand(j)   satisfy demand at market j ;

  cost ..        z  =e=  sum((i,j), c(i,j)*x(i,j)) ;

  supply(i) ..   sum(j, x(i,j))  =l=  a(i) ;

  demand(j) ..   sum(i, x(i,j))  =g=  b(j) ;

  Model transport /all/ ;

  Solve transport using lp minimizing z ;

  Display x.l, x.m ;

$log Printing modelstat constants to the log
$log %ModelStat.Optimal                 %
$log %ModelStat.Locally Optimal         %
$log %ModelStat.Unbounded               %
$log %ModelStat.Infeasible              %
$log %ModelStat.Locally Infeasible      %
$log %ModelStat.Intermediate Infeasible %
$log %ModelStat.Feasible Solution       %
$log %ModelStat.Integer Solution        %
$log %ModelStat.Intermediate Non-Integer%
$log %ModelStat.Integer Infeasible      %
$log %ModelStat.Licensing Problem       %
$log %ModelStat.Error Unknown           %
$log %ModelStat.Error No Solution       %
$log %ModelStat.No Solution Returned    %
$log %ModelStat.Solved Unique           %
$log %ModelStat.Solved                  %
$log %ModelStat.Solved Singular         %
$log %ModelStat.Unbounded - No Solution %
$log %ModelStat.Infeasible - No Solution%
