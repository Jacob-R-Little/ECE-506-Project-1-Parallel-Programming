#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include "data.h"
#include "timer.h"

int main(int argc, char **argv)
{
    if(argc != 3){
        printf("ERROR: Please specify only 2 files.\n");
        exit(EXIT_FAILURE);
    }
        
    struct timespec start;
    start_timer(&start);

    data_struct *d_1 = get_data_struct(argv[1]);
    data_struct *d_2 = get_data_struct(argv[2]);
    data_struct *d_3 = malloc(sizeof(data_struct));
    
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
    //d_3->data_point = calloc(d_1->rows, sizeof(double));

    start_timer(&start);

    // calculate product
    for (unsigned int i=0; i<d_1->rows; i++) {
        for (unsigned int j=0; j<d_2->cols; j++) {
            for (unsigned int k = 0; k<d_1->cols; k++) {
                d_3->data_point[i][j] += d_1->data_point[i][k] * d_2->data_point[k][j];
            }
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
