CC=gcc
CFLAGS=-Iinclude/ -Wall -std=gnu99 -g
OMPFLAGS=$(CFLAGS) -fopenmp
LIBS=src/data_handling.c
TUNE=-O2

.PHONY: all
all: serial omp mpi hybrid

.PHONY: serial
serial:
	$(CC) $(TUNE) $(CFLAGS) -o bin/serial_vecadd $(LIBS) src/serial_vecadd.c
	$(CC) $(TUNE) $(CFLAGS) -o bin/serial_matadd $(LIBS) src/serial_matadd.c
	$(CC) $(TUNE) $(CFLAGS) -o bin/serial_matmul $(LIBS) src/serial_matmul.c

omp: omp_vecadd omp_matadd omp_matmul

omp_vecadd:
	$(CC) $(TUNE) $(OMPFLAGS) -o bin/omp_vecadd $(LIBS) src/omp_vecadd.c
omp_matadd:
	$(CC) $(TUNE) $(OMPFLAGS) -o bin/omp_matadd $(LIBS) src/omp_matadd.c
omp_matmul:
	$(CC) $(TUNE) $(OMPFLAGS) -o bin/omp_matmul $(LIBS) src/omp_matmul.c
			
.PHONY: mpi
mpi: mpi_vecadd mpi_matadd mpi_matmul

mpi_vecadd:
	mpicc $(TUNE) $(CFLAGS) -o bin/mpi_vecadd $(LIBS) src/mpi_vecadd.c
mpi_matadd:
	mpicc $(TUNE) $(CFLAGS) -o bin/mpi_matadd $(LIBS) src/mpi_matadd.c
mpi_matmul:
	mpicc $(TUNE) $(CFLAGS) -o bin/mpi_matmul $(LIBS) src/mpi_matmul.c

.PHONY: hybrid
hybrid: hybrid_vecadd hybrid_matadd hybrid_matmul

hybrid_vecadd:
	mpicc $(TUNE) $(OMPFLAGS) -o bin/hybrid_vecadd $(LIBS) src/hybrid_vecadd.c
hybrid_matadd:
	mpicc $(TUNE) $(OMPFLAGS) -o bin/hybrid_matadd $(LIBS) src/hybrid_matadd.c
hybrid_matmul:
	mpicc $(TUNE) $(OMPFLAGS) -o bin/hybrid_matmul $(LIBS) src/hybrid_matmul.c

.PHONY: clean
clean: clean_serial clean_openmp clean_mpi clean_hybrid

.PHONY: clean_serial
clean_serial:
	@rm -rf bin/serial_vecadd
	@rm -rf bin/serial_matadd
	@rm -rf bin/serial_matmul

.PHONY: clean_openmp
clean_openmp:
	@rm -rf bin/omp_vecadd
	@rm -rf bin/omp_matadd
	@rm -rf bin/omp_matmul

.PHONY: clean_mpi
clean_mpi:
	@rm -rf bin/mpi_vecadd
	@rm -rf bin/mpi_matadd
	@rm -rf bin/mpi_matmul

.PHONY: clean_hybrid
clean_hybrid:
	@rm -rf bin/hybrid_vecadd
	@rm -rf bin/hybrid_matadd
	@rm -rf bin/hybrid_matmul

MAT_A = mat_a.txt
MAT_B = mat_b.txt
MAT_C0 = mat_c0.txt
MAT_C = mat_c.txt

PROBLEM = matmul

SERIAL_BIN = serial_$(PROBLEM)

run_serial: serial
	@echo "*** Running $(SERIAL_BIN) and storing result in $(MAT_C0) ***"
	bin/$(SERIAL_BIN) $(MAT_A) $(MAT_B) > $(MAT_C0)

NUM_THREADS = 8
OMP_BIN = omp_$(PROBLEM)

run_omp:
	@echo "*** Running $(OMP_BIN) and storing result in $(MAT_C) ***"
	bin/$(OMP_BIN) $(NUM_THREADS) $(MAT_A) $(MAT_B) > $(MAT_C)

NUM_PROC = 8
MPI_BIN = mpi_$(PROBLEM)

run_mpi:
	@echo "*** Running $(MPI_BIN) and storing result in $(MAT_C) ***"
	mpirun -np $(NUM_PROC) -ppn 8 bin/$(MPI_BIN) $(MAT_A) $(MAT_B) > $(MAT_C)

HYBRID_BIN = hybrid_$(PROBLEM)

run_hybrid:
	@echo "*** Running $(HYBRID_BIN) and storing result in $(MAT_C) ***"
	mpirun -np $(NUM_PROC) bin/$(HYBRID_BIN) $(NUM_THREADS) $(MAT_A) $(MAT_B) > $(MAT_C)
		
SEED_A = 406
SEED_B = 506
X_A = 3
Y_A = 4
X_B = 4
Y_B = 3
OUT_A = mat_a.txt
OUT_B = mat_b.txt

gen_matrix:
	python3 gen_matrix.py $(SEED_A) $(X_A) $(Y_A) > $(OUT_A)
	python3 gen_matrix.py $(SEED_B) $(X_B) $(Y_B) > $(OUT_B)

# In self-grader we don't run serial each time, correct output is generated only once
# check:
check: run_serial run_omp
	@echo "*** Running diff ***"
# not present on ARC, but you can install it locally with apt install numdiff or try to use diff
# 	numdiff $(MAT_C0) $(MAT_C)
	diff -iw $(MAT_C0) $(MAT_C)

pack:
	zip -r project1.zip src/ include/

JOB = job.serial

sbatch:
	sbatch $(JOB)
