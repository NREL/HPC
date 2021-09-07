$title Multi-period DC-OPF for IEEE 24-bus network considering wind and load shedding

$onText

Example from the book: Soroudi, Alireza. Power System Optimization Modeling in GAMS. Springer, 2017.
                       DOI: doi.org/10.1007/978-3-319-62350-4
For more details on the problem formulation please refer to Chapter 6 of the book.

$offText

Set
   bus        / 1*24   /
   slack(bus) / 13     /
   Gen        / g1*g12 /
   t          / t1*t24 /;

Scalar
   Sbase /   100 /
   VOLL  / 10000 /
   VOLW  /    50 /;

Alias (bus,node);

Table GD(Gen,*) 'generating units characteristics'
        Pmax Pmin   b     CostsD costst RU   RD   SU   SD   UT   DT   uini U0  So
   g1   400  100    5.47  0      0      47   47   105  108  1    1    1    5   0
   g2   400  100    5.47  0      0      47   47   106  112  1    1    1    6   0
   g3   152  30.4   13.32 1430.4 1430.4 14   14   43   45   8    4    1    2   0
   g4   152  30.4   13.32 1430.4 1430.4 14   14   44   57   8    4    1    2   0
   g5   155  54.25  16    0      0      21   21   65   77   8    8    0    0   2
   g6   155  54.25  10.52 312    312    21   21   66   73   8    8    1    10  0
   g7   310  108.5  10.52 624    624    21   21   112  125  8    8    1    10  0
   g8   350  140    10.89 2298   2298   28   28   154  162  8    8    1    5   0
   g9   350  75     20.7  1725   1725   49   49   77   80   8    8    0    0   2
   g10  591  206.85 20.93 3056.7 3056.7 21   21   213  228  12   10   0    0   8
   g11  60   12     26.11 437    437    7    7    19   31   4    2    0    0   1
   g12  300  0      0     0      0      35   35   315  326  0    0    1    2   0;
* -----------------------------------------------------

Set GB(bus,Gen) 'connectivity index of each generating unit to each bus'
/
   18.g1
   21.g2
   1. g3
   2. g4
   15.g5
   16.g6
   23.g7
   23.g8
   7. g9
   13.g10
   15.g11
   22.g12
/;

Table BusData(bus,*) 'demands of each bus in MW'
       Pd   Qd
   1   108  22
   2   97   20
   3   180  37
   4   74   15
   5   71   14
   6   136  28
   7   125  25
   8   171  35
   9   175  36
   10  195  40
   13  265  54
   14  194  39
   15  317  64
   16  100  20
   18  333  68
   19  181  37
   20  128  26;
****************************************************

Table branch(bus,node,*) 'network technical characteristics'
           r        x        b       limit
   1 .2    0.0026   0.0139   0.4611  175
   1 .3    0.0546   0.2112   0.0572  175
   1 .5    0.0218   0.0845   0.0229  175
   2 .4    0.0328   0.1267   0.0343  175
   2 .6    0.0497   0.192    0.052   175
   3 .9    0.0308   0.119    0.0322  175
   3 .24   0.0023   0.0839   0       400
   4 .9    0.0268   0.1037   0.0281  175
   5 .10   0.0228   0.0883   0.0239  175
   6 .10   0.0139   0.0605   2.459   175
   7 .8    0.0159   0.0614   0.0166  175
   8 .9    0.0427   0.1651   0.0447  175
   8 .10   0.0427   0.1651   0.0447  175
   9 .11   0.0023   0.0839   0       400
   9 .12   0.0023   0.0839   0       400
   10.11   0.0023   0.0839   0       400
   10.12   0.0023   0.0839   0       400
   11.13   0.0061   0.0476   0.0999  500
   11.14   0.0054   0.0418   0.0879  500
   12.13   0.0061   0.0476   0.0999  500
   12.23   0.0124   0.0966   0.203   500
   13.23   0.0111   0.0865   0.1818  500
   14.16   0.005    0.0389   0.0818  500
   15.16   0.0022   0.0173   0.0364  500
   15.21   0.00315  0.0245   0.206   1000
   15.24   0.0067   0.0519   0.1091  500
   16.17   0.0033   0.0259   0.0545  500
   16.19   0.003    0.0231   0.0485  500
   17.18   0.0018   0.0144   0.0303  500
   17.22   0.0135   0.1053   0.2212  500
   18.21   0.00165  0.01295  0.109   1000
   19.20   0.00255  0.0198   0.1666  1000
   20.23   0.0014   0.0108   0.091   1000
   21.22   0.0087   0.0678   0.1424  500 ;
* ----------------------------------------------

Table WD(t,*)
        w                   d
   t1   0.0786666666666667  0.684511335492475
   t2   0.0866666666666667  0.644122690036197
   t3   0.117333333333333   0.61306915602972
   t4   0.258666666666667   0.599733282530006
   t5   0.361333333333333   0.588874071251667
   t6   0.566666666666667   0.5980186702229
   t7   0.650666666666667   0.626786054486569
   t8   0.566666666666667   0.651743189178891
   t9   0.484               0.706039245570585
   t10  0.548               0.787007048961707
   t11  0.757333333333333   0.839016955610593
   t12  0.710666666666667   0.852733854067441
   t13  0.870666666666667   0.870642027052772
   t14  0.932               0.834254143646409
   t15  0.966666666666667   0.816536483139646
   t16  1                   0.819394170318156
   t17  0.869333333333333   0.874071251666984
   t18  0.665333333333333   1
   t19  0.656               0.983615926843208
   t20  0.561333333333333   0.936368832158506
   t21  0.565333333333333   0.887597637645266
   t22  0.556               0.809297008954087
   t23  0.724               0.74585635359116
   t24  0.84                0.733473042484283;

Parameter Wcap(bus) / 8 200, 19 150, 21 100 /;

branch(bus,node,'x')$(branch(bus,node,'x')=0)         =   branch(node,bus,'x');
branch(bus,node,'Limit')$(branch(bus,node,'Limit')=0) =   branch(node,bus,'Limit');
branch(bus,node,'bij')$branch(bus,node,'Limit')       = 1/branch(bus,node,'x');
branch(bus,node,'z')$branch(bus,node,'Limit')         = sqrt(sqr(branch(bus,node,'x')) + sqr(branch(bus,node,'r')));
branch(node,bus,'z')                                  = branch(bus,node,'z');

Parameter conex(bus,node);
conex(bus,node)$(branch(bus,node,'limit') and branch(node,bus,'limit')) = 1;
conex(bus,node)$(conex(node,bus)) = 1;

Variable OF, Pij(bus,node,t), Pg(Gen,t), delta(bus,t), lsh(bus,t), Pw(bus,t), pc(bus,t);
Equation const1, const2, const3, const4, const5, const6;

const1(bus,node,t)$(conex(bus,node))..
   Pij(bus,node,t) =e= branch(bus,node,'bij')*(delta(bus,t) - delta(node,t));

const2(bus,t)..
   lsh(bus,t)$BusData(bus,'pd') + Pw(bus,t)$Wcap(bus) + sum(Gen$GB(bus,Gen), Pg(Gen,t)) - WD(t,'d')*BusData(bus,'pd')/Sbase =e= sum(node$conex(node,bus), Pij(bus,node,t));

const3..
   OF =g= sum((bus,Gen,t)$GB(bus,Gen), Pg(Gen,t)*GD(Gen,'b')*Sbase) + sum((bus,t), VOLL*lsh(bus,t)*Sbase$BusData(bus,'pd') + VOLW*Pc(bus,t)*sbase$Wcap(bus));

const4(gen,t)..
   pg(gen,t+1) - pg(gen,t) =l= GD(gen,'RU')/Sbase;

const5(gen,t)..
   pg(gen,t-1) - pg(gen,t) =l= GD(gen,'RD')/Sbase;

const6(bus,t)$Wcap(bus)..
   pc(bus,t) =e= WD(t,'w')*Wcap(bus)/Sbase - pw(bus,t);

Model opf / const1, const2, const3, const4, const5, const6 /;

Pg.lo(Gen,t) = GD(Gen,'Pmin')/Sbase;
Pg.up(Gen,t) = GD(Gen,'Pmax')/Sbase;

delta.up(bus,t)   = pi/2;
delta.lo(bus,t)   =-pi/2;
delta.fx(slack,t) = 0;

Pij.up(bus,node,t)$((conex(bus,node))) = 1*branch(bus,node,'Limit')/Sbase;
Pij.lo(bus,node,t)$((conex(bus,node))) =-1*branch(bus,node,'Limit')/Sbase;
lsh.up(bus,t) = WD(t,'d')*BusData(bus,'pd')/Sbase;
lsh.lo(bus,t) = 0;

Pw.up(bus,t)  = WD(t,'w')*Wcap(bus)/Sbase;
Pw.lo(bus,t)  = 0;
Pc.up(bus,t)  = WD(t,'w')*Wcap(bus)/Sbase;
Pc.lo(bus,t)  = 0;

solve opf minimizing OF using lp;

*$onText
Parameter report(t,bus,*), Congestioncost, lmp(bus,t);
report(t,bus,'Gen(MW)')    = 1*sum(Gen$GB(bus,Gen), Pg.l(Gen,t))*sbase;
report(t,bus,'Angle')      = delta.l(bus,t);
report(t,bus,'LSH')        = lsh.l(bus,t)*sbase ;
report(t,bus,'LMP($/MWh)') = const2.m(bus,t)/sbase;
report(t,bus,'Wind(MW)')   = Pw.l(bus,t)/sbase;
report(t,bus,'Curtailment(MW)') = Pc.l(bus,t)/sbase;
Congestioncost = sum((bus,node,t)$conex(bus,node), Pij.l(bus,node,t)*(-const2.m(bus,t) + const2.m(node,t)))/2;
lmp(bus,t) = report(t,bus,'LMP($/MWh)');
display report, Pij.l, Congestioncost;

$ifI %system.fileSys%==Unix $exit
$call MSAppAvail Excel
$ifThen not errorLevel 1
   execute_unload "opf.gdx" report
   execute 'gdxxrw.exe  OPF.gdx o=OPF.xls par=report rng=report!A1'
   execute_unload "opf.gdx" Pw.l
   execute 'gdxxrw.exe  OPF.gdx o=OPF.xls var=Pw rng=PW!A1'
   execute_unload "opf.gdx" Pc.l
   execute 'gdxxrw.exe  OPF.gdx o=OPF.xls var=Pc rng=PC!A1'
   execute_unload "opf.gdx" Pg.l
   execute 'gdxxrw.exe  OPF.gdx o=OPF.xls var=Pg rng=Pg!A1'
   execute_unload "opf.gdx" LSH.l
   execute 'gdxxrw.exe  OPF.gdx o=OPF.xls var=LSH rng=lsh!A1'
   execute_unload "opf.gdx" Lmp
   execute 'gdxxrw.exe  OPF.gdx o=OPF.xls par=Lmp rng=lmp!A1'
$endIf
*$offText