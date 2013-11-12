mic-apps
========

MIC Applications

Xeon Phi uses traditional programming models like OpenMP/Pthreads/MPI. The expectation is that re-compiling the code with the -mmic option will achieve high performance. Our experience shows that, porting legacy code or developing new code still needs a lot of developer interventions. For example, we need to manually vectorize the code and expose parallelism. In this project, we try to collect the code that needs such an effort.

Jianbin Fang
--
