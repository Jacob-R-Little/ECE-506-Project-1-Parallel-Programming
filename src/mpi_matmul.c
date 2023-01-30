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
    int num_rank_itemsA;
    int num_rank_itemsB;
    int num_rank_results;

    int rowsA, colsA;
    int rowsB, colsB;
    int rowsC, colsC;
    int num_rank_rowsA;

    double * MPI_send_buff_A = NULL;
    double * MPI_send_buff_B = NULL;
    double * MPI_rank_buff_A = NULL;
    double * MPI_rank_buff_C = NULL;
    double * MPI_receive_buff = NULL;

    struct timespec start;
    data_struct *d_1 = NULL;
    data_struct *d_2 = NULL;
    data_struct *d_3 = NULL;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == RANK0) {
        start_timer(&start);

        d_1 = get_data_struct(argv[1]);
        d_2 = get_data_struct(argv[2]);
        d_3 = malloc(sizeof(data_struct));

        stop_timer(&start);
        fprintf(stderr, " (reading input)\n");

        if(d_1->cols != d_2->rows){
            printf("ERROR: Matrix dimension mismatch.\n");
            exit(EXIT_FAILURE);
        }

        // allocate data for sum array
        d_3->rows = d_1->rows;
        d_3->cols = d_2->cols;
        d_3->data_point = calloc(d_3->rows, sizeof(double*));
        for (unsigned int i=0; i<d_3->rows; i++) {
            d_3->data_point[i] = calloc(d_3->cols, sizeof(double));
        }

        start_timer(&start);

        rowsA = d_1->rows; colsA = d_1->cols;
        rowsB = d_2->rows; colsB = d_2->cols;
        rowsC = d_3->rows; colsC = d_3->cols;

        num_rank_itemsA = (rowsA * colsA) / world_size;
        num_rank_itemsB = rowsB * colsB;
        num_rank_results = (rowsC * colsC) / world_size;
        num_rank_rowsA = rowsA / world_size;

        MPI_send_buff_A = calloc(rowsA * colsA, sizeof(double));
        MPI_send_buff_B = calloc(rowsB * colsB, sizeof(double));
        MPI_receive_buff = calloc(rowsC * colsC, sizeof(double));

        for (unsigned int i=0; i<rowsA; i++) {
            for (unsigned int j=0; j<colsA; j++) {
                MPI_send_buff_A[i * colsA + j] = d_1->data_point[i][j];
            }
        }

        for (unsigned int i=0; i<rowsB; i++) {
            for (unsigned int j=0; j<colsB; j++) {
                MPI_send_buff_B[i * colsB + j] = d_2->data_point[i][j];
            }
        }

    }

    MPI_Bcast(&rowsA, 1, MPI_INT, RANK0, MPI_COMM_WORLD);
    MPI_Bcast(&colsA, 1, MPI_INT, RANK0, MPI_COMM_WORLD);
    MPI_Bcast(&rowsB, 1, MPI_INT, RANK0, MPI_COMM_WORLD);
    MPI_Bcast(&colsB, 1, MPI_INT, RANK0, MPI_COMM_WORLD);
    MPI_Bcast(&rowsC, 1, MPI_INT, RANK0, MPI_COMM_WORLD);
    MPI_Bcast(&colsC, 1, MPI_INT, RANK0, MPI_COMM_WORLD);

    MPI_Bcast(&num_rank_itemsA, 1, MPI_INT, RANK0, MPI_COMM_WORLD);
    MPI_Bcast(&num_rank_itemsB, 1, MPI_INT, RANK0, MPI_COMM_WORLD);
    MPI_Bcast(&num_rank_results, 1, MPI_INT, RANK0, MPI_COMM_WORLD);
    MPI_Bcast(&num_rank_rowsA, 1, MPI_INT, RANK0, MPI_COMM_WORLD);

    MPI_rank_buff_A = calloc(num_rank_itemsA, sizeof(double));
    MPI_rank_buff_C = calloc(num_rank_results, sizeof(double));

    if (world_rank != 0) {
        MPI_send_buff_B = calloc(num_rank_itemsB, sizeof(double));
    }

    MPI_Bcast(MPI_send_buff_B, num_rank_itemsB, MPI_DOUBLE, RANK0, MPI_COMM_WORLD);
    MPI_Scatter(MPI_send_buff_A, num_rank_itemsA, MPI_DOUBLE, MPI_rank_buff_A, num_rank_itemsA, MPI_DOUBLE, RANK0, MPI_COMM_WORLD);

    for (unsigned int i=0; i<num_rank_rowsA; i++) {
        for (unsigned int j=0; j<colsB; j++) {
            for (unsigned int k = 0; k<colsA; k++) {
                MPI_rank_buff_C[i * colsC + j] += MPI_rank_buff_A[i * colsA + k] * MPI_send_buff_B[k * colsB + j];
            }
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(MPI_rank_buff_C, num_rank_results, MPI_DOUBLE, MPI_receive_buff, num_rank_results, MPI_DOUBLE, RANK0, MPI_COMM_WORLD);

    if (world_rank == RANK0) {

        for (unsigned int i=0; i<rowsC; i++) {
            for (unsigned int j=0; j<colsC; j++) {
                d_3->data_point[i][j] = MPI_receive_buff[i * colsC + j];
            }
        }

        stop_timer(&start);
        
	    fprintf(stderr, " (calculating answer)\n");

        start_timer(&start);
        
        // print output
        for (unsigned int i=0; i<d_3->rows; i++) {
            for (unsigned int j=0; j<d_3->cols; j++) {
                printf("%f ",d_3->data_point[i][j]);
            }
            printf("\n");
        }

        stop_timer(&start);
        fprintf(stderr, " (printing output)\n");
    }

    MPI_Finalize();

}
