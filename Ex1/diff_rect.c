#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>



int find_sqrt_procs(int procs){
    for(int i=0;i<procs;i++){
        if(procs==i*i){
            return i;
        }
    }
    return 0;




}

typedef struct Diagnostics_s
{
    double time;
    double heat;
} Diagnostics;

typedef struct Diffusion2D_s
{
    double D_, L_, T_;
    int N_, Ntot_, real_N_;
    double dr_, dt_, fac_;
    int rank_, procs_,sqrt_procs_;
    int local_N_;
    double *rho_, *rho_tmp_;
    Diagnostics *diag_;
} Diffusion2D;

void initialize_density(Diffusion2D *D2D)
{
    int real_N_ = D2D->real_N_;
    int N_ = D2D->N_;
    int local_N_ = D2D->local_N_;
    double *rho_ = D2D->rho_;
    double dr_ = D2D->dr_;
    double L_ = D2D->L_;
    int rank_ = D2D->rank_;
    //int procs_ = D2D->procs_;
    int sqrt_procs_=D2D->sqrt_procs_;
    int gi,gj;

    /// Initialize rho(x, y, t=0).
    double bound = 0.25 * L_;

    for (int i = 1; i <= local_N_; ++i) {
        gi = (rank_/sqrt_procs_) * (N_ / sqrt_procs_) + i; // convert local i index to global index
        // convert local j index to global index
        for (int j = 1; j <= N_; ++j) {
            gj = (rank_ % sqrt_procs_) * (N_ / sqrt_procs_) + j; 
            if (fabs((gi - 1)*dr_ - 0.5*L_) < bound && fabs((gj-1)*dr_ - 0.5*L_) < bound) {
                rho_[i*real_N_ + j] = 1;
            } else {
                rho_[i*real_N_ + j] = 0;
            }
        }
    }
}

void init(Diffusion2D *D2D,
                const double D,
                const double L,
                const int N,
                const int T,
                const double dt,
                const int rank,
                const int procs,
                int sqrt_procs)
{
    D2D->D_ = D;
    D2D->L_ = L;
    D2D->N_ = N;
    D2D->T_ = T;
    D2D->dt_ = dt;
    D2D->rank_ = rank;
    D2D->procs_ = procs;
    D2D->sqrt_procs_=sqrt_procs;

    // Real space grid spacing.
    D2D->dr_ = D2D->L_ / (D2D->N_ - 1);

    // Stencil factor.
    D2D->fac_ = D2D->dt_ * D2D->D_ / (D2D->dr_ * D2D->dr_);

    // Number of rows/columns per process.
    D2D->local_N_ = D2D->N_ / D2D->sqrt_procs_;

    // Small correction for the last process.
    if (D2D->rank_ == D2D->procs_ - 1)
        D2D->local_N_ += D2D->N_ % D2D->sqrt_procs_;

    // Actual dimension of a row/column (+2 for the ghost cells).
    D2D->real_N_ = D2D->local_N_ + 2;

    // Total number of cells.
    D2D->Ntot_ = (D2D->local_N_ + 2) * (D2D->local_N_ + 2);

    D2D->rho_ = (double *)calloc(D2D->Ntot_, sizeof(double));
    D2D->rho_tmp_ = (double *)calloc(D2D->Ntot_, sizeof(double));
    D2D->diag_ = (Diagnostics *)calloc(D2D->T_, sizeof(Diagnostics));

    // Check that the timestep satisfies the restriction for stability.
    if (D2D->rank_ == 0)
        printf("timestep from stability condition is %.10lf\n", D2D->dr_ * D2D->dr_ / (4.0 * D2D->D_));

    initialize_density(D2D);
}

void advance(Diffusion2D *D2D)
{
    //int N_ = D2D->N_;
    int real_N_ = D2D->real_N_;
    int local_N_ = D2D->local_N_;
    double *rho_ = D2D->rho_;
    double *rho_tmp_ = D2D->rho_tmp_;
    double fac_ = D2D->fac_;
    int rank_ = D2D->rank_;
    //int procs_ = D2D->procs_;
    int sqrt_procs_ = D2D->sqrt_procs_;
    double *left_send_buff = (double *)calloc(real_N_-2, sizeof(double));
    double *right_send_buff = (double *)calloc(real_N_-2, sizeof(double));
    double *left_recv_buff = (double *)calloc(real_N_-2, sizeof(double));
    double *right_recv_buff = (double *)calloc(real_N_-2, sizeof(double));   

	// Non-blocking MPI
	MPI_Request req[8];
	MPI_Status status[8];

    //int prev_rank = rank_ - 1;
    //int next_rank = rank_ + 1;

    // Exchange ALL necessary ghost cells with neighboring ranks.#########################################################################################################
    if (rank_ / sqrt_procs_ != 0) {//check if top super-row of array
        // TODO:MPI
		MPI_Irecv(&rho_[           0*real_N_], local_N_+2, MPI_DOUBLE, rank_-sqrt_procs_, 100, MPI_COMM_WORLD, &req[0]);
		MPI_Isend(&rho_[           1*real_N_], local_N_+2, MPI_DOUBLE, rank_-sqrt_procs_, 100, MPI_COMM_WORLD, &req[1]);
    }
    else {
        // the purpose of this part will become
        // clear when using asynchronous communication.
		req[0] = MPI_REQUEST_NULL;
		req[1] = MPI_REQUEST_NULL;
    }

    if (rank_ / sqrt_procs_ != sqrt_procs_ -1) { //check if bottom super-row of array
        // TODO:MPI
		MPI_Irecv(&rho_[(real_N_-1)*real_N_], local_N_+2, MPI_DOUBLE, rank_+sqrt_procs_, 100, MPI_COMM_WORLD, &req[2]);
		MPI_Isend(&rho_[(real_N_-2)*real_N_], local_N_+2, MPI_DOUBLE, rank_+sqrt_procs_, 100, MPI_COMM_WORLD, &req[3]);
    }
    else {
        // the purpose of this part will become 
        // clear when using asynchronous communication.
		req[2] = MPI_REQUEST_NULL;
		req[3] = MPI_REQUEST_NULL;
    }

    if (rank_ % sqrt_procs_ != 0) { //check if left super-column of array
        // TODO:MPI
        int k=0;
        for(k=2;k<real_N_-2;k++){
            left_send_buff[k-2]=rho_[k*real_N_+1];
        }
		MPI_Irecv(&left_recv_buff[           0], local_N_-2, MPI_DOUBLE, rank_-1, 100, MPI_COMM_WORLD, &req[4]);
		MPI_Isend(&left_send_buff[           0], local_N_-2, MPI_DOUBLE, rank_-1, 100, MPI_COMM_WORLD, &req[5]);
    }
    else {
        // the purpose of this part will become 
        // clear when using asynchronous communication.
		req[4] = MPI_REQUEST_NULL;
		req[5] = MPI_REQUEST_NULL;
    }  

    if (rank_ % sqrt_procs_ != sqrt_procs_-1) { //check if right super-column of array
        // TODO:MPI
        int k=0;
        for(k=2;k<real_N_-2;k++){
            right_send_buff[k-2]=rho_[k*real_N_+real_N_-2];
        }
		MPI_Irecv(&right_recv_buff[          0], local_N_-2, MPI_DOUBLE, rank_+1, 100, MPI_COMM_WORLD, &req[6]);
		MPI_Isend(&right_send_buff[          0], local_N_-2, MPI_DOUBLE, rank_+1, 100, MPI_COMM_WORLD, &req[7]);
    }
    else {
        // the purpose of this part will become 
        // clear when using asynchronous communication.
		req[6] = MPI_REQUEST_NULL;
		req[7] = MPI_REQUEST_NULL;
    }  
    // Central differences in space, forward Euler in time with Dirichlet
    // boundaries.
    //printf("N_ %d\n",N_);
    //printf("local_N_ %d\n", local_N_);
    //calculate inner cells #############################################################################################################################################
    for (int i = 2; i < local_N_; ++i) { //i is for rows , j for collumns  //this "for" goes from the second row of the batch to the one before the last
                                         // local_N_ is column size for batch (total size/procs, for size=1024 and procs=4 local_N_ is 256)
        for (int j = 2; j < local_N_; ++j) {  //N_ = 1024 or simulated row size //real_N_ = 1026 or row size including ghost cells?
            rho_tmp_[i*real_N_ + j] = rho_[i*real_N_ + j] + //real_N_ is row size
                                     fac_
                                     *
                                     (
                                     + rho_[i*real_N_ + (j+1)]
                                     + rho_[i*real_N_ + (j-1)]
                                     + rho_[(i+1)*real_N_ + j]
                                     + rho_[(i-1)*real_N_ + j]
                                     - 4.*rho_[i*real_N_ + j]
                                     );
        }
    }

	// ensure boundaries have arrived #####################################################################################################################################
	MPI_Waitall(4, req, status);

    // fill left and right to-be-received columns from received-buffers
    if(rank_ % sqrt_procs_ != 0){
    int k=0;
        for(k=2;k<real_N_-2;k++){
            rho_[k*real_N_]=left_recv_buff[k-2];
        }
    }
    if(rank_ % sqrt_procs_ != sqrt_procs_-1){
    int k=0;
        for(k=2;k<real_N_-2;k++){
            rho_[k*real_N_+real_N_-2]=right_recv_buff[k-2];
        }
    }
    // Update the first and the last rows of each rank.
    for (int i = 1; i <= local_N_; i += local_N_- 1) {
        for (int j = 1; j <= local_N_; ++j) {
            rho_tmp_[i*real_N_ + j] = rho_[i*real_N_ + j] +
                                     fac_
                                     *
                                     (
                                     + rho_[i*real_N_ + (j+1)]
                                     + rho_[i*real_N_ + (j-1)]
                                     + rho_[(i+1)*real_N_ + j]
                                     + rho_[(i-1)*real_N_ + j]
                                     - 4.*rho_[i*real_N_ + j]
                                     );
        }
    }
    //update left and right column of each rank
    for (int i = 1; i <= local_N_; ++i) {
        for (int j = 1; j <= local_N_; j += local_N_- 1){
            rho_tmp_[i*real_N_ + j] = rho_[i*real_N_ + j] +
                                     fac_
                                     *
                                     (
                                     + rho_[i*real_N_ + (j+1)]
                                     + rho_[i*real_N_ + (j-1)]
                                     + rho_[(i+1)*real_N_ + j]
                                     + rho_[(i-1)*real_N_ + j]
                                     - 4.*rho_[i*real_N_ + j]
                                     );
        }
    }


    //#####################################################################################################################################################################
    // Swap rho_ with rho_tmp_. This is much more efficient,
    // because it does not copy element by element, just replaces storage
    // pointers.
    double *tmp_ = D2D->rho_tmp_;
    D2D->rho_tmp_ = D2D->rho_;
    D2D->rho_ = tmp_;
}

void compute_diagnostics(Diffusion2D *D2D, const int step, const double t)
{
    int N_ = D2D->N_;
    int real_N_ = D2D->real_N_;
    int local_N_ = D2D->local_N_;
    double *rho_ = D2D->rho_;
    double dr_ = D2D->dr_;
    int rank_ = D2D->rank_;

    double heat = 0.0;
    for(int i = 1; i <= local_N_; ++i)
        for(int j = 1; j <= N_; ++j)
            heat += rho_[i*real_N_ + j] * dr_ * dr_;

    // TODO:MPI, reduce heat (sum)
    MPI_Reduce(rank_ == 0? MPI_IN_PLACE: &heat, &heat, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); 


    if (rank_ == 0) {
#if DEBUG
        printf("t = %lf heat = %lf\n", t, heat);
#endif
        D2D->diag_[step].time = t;
        D2D->diag_[step].heat = heat;
    }
}

void write_diagnostics(Diffusion2D *D2D, const char *filename)
{

    FILE *out_file = fopen(filename, "w");
    for (int i = 0; i < D2D->T_; i++)
        fprintf(out_file, "%f\t%f\n", D2D->diag_[i].time, D2D->diag_[i].heat);
    fclose(out_file);
}


int main(int argc, char* argv[])
{
    if (argc < 1) {
        printf("Usage: %s D L T N dt\n", argv[0]);
        return 1;
    }

    int rank, procs, sqrt_procs=0 ;
    //TODO:MPI Initialize MPI, number of ranks (rank) and number of processes (nprocs) involved in the communicator
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    sqrt_procs=find_sqrt_procs(procs);
    //printf("our sqrt of procs is %d",sqrt_procs);


    if(sqrt_procs==0){
        printf("Process count should be a power of 4\n");
        return 1;  
    }
    const double D = atof(argv[1]);
    const double L = atoi(argv[2]);
    const int N = atoi(argv[3]);
    const int T = atoi(argv[4]);
    const double dt = atof(argv[5]);

    Diffusion2D system;

    init(&system, D, L, N, T, dt, rank, procs,sqrt_procs); //creates on system for each rank representing what each rank sees

    double t0 = MPI_Wtime();
    for (int step = 0; step < T; ++step) {
        advance(&system);
#ifndef _PERF_
        compute_diagnostics(&system, step, dt * step);
#endif
    }
    double t1 = MPI_Wtime();

    if (rank == 0)
        printf("Timing: %d %lf\n", N, t1-t0);

#ifndef _PERF_
    if (rank == 0) {
        char diagnostics_filename[256];
        sprintf(diagnostics_filename, "diagnostics_mpi_%d.dat", procs);
        write_diagnostics(&system, diagnostics_filename);
    }
#endif

    MPI_Finalize();
    return 0;
}
