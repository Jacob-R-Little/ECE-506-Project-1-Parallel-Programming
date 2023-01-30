#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <mpi.h>
#include <string.h>
#include "data.h"
#include "timer.h"

#define RANK0 0

int main(int argc, char *argv[]) {
    
    if(argc != 3){
        printf("ERROR: Please specify only 2 files.\n");
        exit(EXIT_FAILURE);
    }

    int world_size;
    int world_rank;
    int num_rank_items = 0;
    int num_total_items = 0;

    double * MPI_send_buff_A = NULL;
    double * MPI_send_buff_B = NULL;
    double * MPI_rank_buff_A = NULL;
    double * MPI_rank_buff_B = NULL;
    double * MPI_rank_buff_C = NULL;
    double * MPI_receive_buff = NULL;
    
    struct timespec start;
    data_struct *d_1 = NULL;
    data_struct *d_2 = NULL;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == RANK0) {
        start_timer(&start);

        d_1 = get_data_struct(argv[1]);
        d_2 = get_data_struct(argv[2]);
        
        stop_timer(&start);
        fprintf(stderr, " (reading input)\n");

        if(d_1->cols != d_2->cols || d_1->rows != d_2->rows){
            printf("ERROR: The number of columns/rows of matrix A must match the number of columns/rows of matrix B.\n");
            printf("num rows %d num cols %d", d_1->rows, d_1->cols);
            exit(EXIT_FAILURE);
        }

        start_timer(&start);

        num_total_items = d_1->rows * d_1->cols;
        num_rank_items = num_total_items / world_size;

        MPI_send_buff_A = malloc(sizeof(double) * num_total_items);
        MPI_send_buff_B = malloc(sizeof(double) * num_total_items);
        MPI_receive_buff = malloc(sizeof(double) * num_total_items);

        for (unsigned int i=0; i<d_1->rows; i++) {
            for (unsigned int j=0; j<d_1->cols; j++) {
                MPI_send_buff_A[i * d_1->cols + j] = d_1->data_point[i][j];
                MPI_send_buff_B[i * d_1->cols + j] = d_2->data_point[i][j];
            }
        }
    }
    
    MPI_Bcast(&num_total_items, 1, MPI_INT, RANK0, MPI_COMM_WORLD);
    MPI_Bcast(&num_rank_items, 1, MPI_INT, RANK0, MPI_COMM_WORLD);
    
    MPI_rank_buff_A = malloc(sizeof(double) * num_rank_items);
    MPI_rank_buff_B = malloc(sizeof(double) * num_rank_items);
    MPI_rank_buff_C = malloc(sizeof(double) * num_rank_items);
    

    MPI_Scatter(MPI_send_buff_A, num_rank_items, MPI_DOUBLE, MPI_rank_buff_A, num_rank_items, MPI_DOUBLE, RANK0, MPI_COMM_WORLD);
    MPI_Scatter(MPI_send_buff_B, num_rank_items, MPI_DOUBLE, MPI_rank_buff_B, num_rank_items, MPI_DOUBLE, RANK0, MPI_COMM_WORLD);
    
    // calculate sum
    for (unsigned int i=0; i<num_rank_items; i++) {
        MPI_rank_buff_C[i] = MPI_rank_buff_A[i] + MPI_rank_buff_B[i];
    }

    MPI_Gather(MPI_rank_buff_C, num_rank_items, MPI_DOUBLE, MPI_receive_buff, num_rank_items, MPI_DOUBLE, RANK0, MPI_COMM_WORLD);

    if (world_rank == RANK0) {

        for (unsigned int i=0; i<d_1->rows; i++) {
            for (unsigned int j=0; j<d_1->cols; j++) {
                d_1->data_point[i][j] = MPI_receive_buff[i * d_1->cols + j];
            }
        }

        stop_timer(&start);

        fprintf(stderr, " (calculating answer)\n");
        
        start_timer(&start);
        
        // print output
        for (unsigned int i=0; i<d_1->rows; i++) {
            for (unsigned int j=0; j<d_1->cols; j++) {
                printf("%f ",d_1->data_point[i][j]);
            }
            printf("\n");
        }

        stop_timer(&start);
        fprintf(stderr, " (printing output)\n");
    }

    MPI_Finalize();
   
}
