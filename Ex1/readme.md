to compile : "mpicc diff_rect.c -o test"
to run : "mpiexec -n 1 ./test 1 1 1024 1000 0.00000001 --mca opal_warn_on_missing_libcuda 0"
to run : "mpiexec -n 4 ./test 1 1 1024 1000 0.00000001 --mca opal_warn_on_missing_libcuda 0"
