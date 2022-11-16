to compile : "mpicc diff_rect.c -o test"
to run : "mpiexec -n 4 ./test --mca opal_warn_on_missing_libcuda 0"
to run : "mpiexec -n 1 ./test --mca opal_warn_on_missing_libcuda 0"
