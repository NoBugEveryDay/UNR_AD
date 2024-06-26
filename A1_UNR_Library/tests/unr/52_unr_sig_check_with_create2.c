#include <unr.h>
#include <uru/sys/realloc.h>
#include <uru/sys/time.h>
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define N 1000
#define M 50

size_t mem_size[N];
void* send_buf[N];
void* recv_buf[N];
void* verf_buf[N];
unr_sig_h send_sig[N], recv_sig[N], rmt_sig[N];
unr_sig_h dup_send_sig[N][M], dup_rmt_sig[N][M];
size_t blk_offset[N][M], blk_size[N][M];
size_t part_offset[N][M], part_size[N][M];
unr_mem_h send_mem_h[N], recv_mem_h[N];
unr_blk_h send_blk_h[N][M], recv_blk_h[N][M], rmt_blk_h[N][M];
unr_plan_h plan_h[N];

int main(int argc, char** argv) {
    printf("N=%d, M=%d\n", N, M);

    int provided, ret;
    ret = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (ret != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Init_thread failed\n");
        exit(EXIT_FAILURE);
    }
    MPI_Query_thread(&provided);
    if (provided != MPI_THREAD_MULTIPLE) {
        fprintf(stderr, "MPI was not initialized with MPI_THREAD_MULTIPLE\n");
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    unr_init();
    srand(0);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) {
        fprintf(stderr, "This test requires exactly 2 processes\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < N; i++) {
        unr_sig_create(&send_sig[i], M);
        unr_sig_create(&recv_sig[i], M);
    }
    MPI_Sendrecv(
        recv_sig, N, MPI_UNR_SIG_H, 1 - rank, 0,
        rmt_sig, N, MPI_UNR_SIG_H, 1 - rank, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            dup_send_sig[i][j] = send_sig[i];
            dup_rmt_sig[i][j] = rmt_sig[i];
        }
    }

    for (int i = 0; i < N; i++) {
        if (i <= 16) {
            mem_size[i] = 1 << i;
        } else {
            mem_size[i] = (rand() % 16) + 1;
        }

        send_buf[i] = uru_calloc(mem_size[i], 1);
        recv_buf[i] = uru_calloc(mem_size[i], 1);
        verf_buf[i] = uru_calloc(mem_size[i], 1);
        if (send_buf[i] == NULL || recv_buf[i] == NULL || verf_buf[i] == NULL) {
            fprintf(stderr, "uru_malloc failed\n");
            exit(1);
        }
        unr_mem_reg(send_buf[i], mem_size[i], &send_mem_h[i]);
        unr_mem_reg(recv_buf[i], mem_size[i], &recv_mem_h[i]);
        for (int j = 0; j < mem_size[i]; j++) {
            char* p = send_buf[i] + j;
            *p = rand() % 100 + 1;
        }

        for (int j = 0; j < M; j++) {
            blk_offset[i][j] = rand() % mem_size[i];
            blk_size[i][j] = rand() % (mem_size[i] - blk_offset[i][j]) + 1;
            unr_blk_reg(send_mem_h[i], blk_offset[i][j], blk_size[i][j], UNR_NO_SIGNAL, UNR_NO_SIGNAL, &send_blk_h[i][j]);
            unr_blk_reg(recv_mem_h[i], blk_offset[i][j], blk_size[i][j], UNR_NO_SIGNAL, UNR_NO_SIGNAL, &recv_blk_h[i][j]);
            part_offset[i][j] = rand() % blk_size[i][j];
            part_size[i][j] = rand() % (blk_size[i][j] - part_offset[i][j]) + 1;
            memcpy(verf_buf[i] + blk_offset[i][j] + part_offset[i][j], send_buf[i] + blk_offset[i][j] + part_offset[i][j], part_size[i][j]);
        }
    }
    unr_mem_reg_sync();
    MPI_Sendrecv(
        recv_blk_h, N * M, MPI_UNR_BLK_H, 1 - rank, 0,
        rmt_blk_h, N * M, MPI_UNR_BLK_H, 1 - rank, 0,
        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    for (int i = 0; i < N; i++) {
        unr_blk_part_send_batch_plan(
            M,
            &send_blk_h[i][0], &dup_send_sig[i][0], &part_offset[i][0], &part_size[i][0],
            &rmt_blk_h[i][0], &dup_rmt_sig[i][0], &part_offset[i][0], &plan_h[i]);
    }

    for (int i = 0; i < N; i++) {
        unr_plan_start(plan_h[i]);
    }

    for (int i = 0; i < N; i++) {
        for (uru_cpu_cyc_t start = uru_cpu_cyc_now(), now = start;
             unr_sig_check(recv_sig[i]) != URU_SIGNAL_TRIGGERED;
             now = uru_cpu_cyc_now()) {
            if (uru_time_diff(start, now) > 10) {
                printf("Timeout!\n");
                return 1;
            }
        }
        for (int j = 0; j < mem_size[i]; j++) {
            volatile char* verf_p = verf_buf[i] + j;
            volatile char* recv_p = recv_buf[i] + j;
            if (*verf_p != *recv_p) {
                printf("Error @ i = %d, j = %d!\n", i, j);
                return 1;
            }
        }
    }

    for (int i = 0; i < N; i++) {
        for (uru_cpu_cyc_t start = uru_cpu_cyc_now(), now = start;
             unr_sig_check(send_sig[i]) != URU_SIGNAL_TRIGGERED;
             now = uru_cpu_cyc_now()) {
            if (uru_time_diff(start, now) > 10) {
                printf("Timeout!\n");
                return 1;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    sleep(1);
    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < N; i++) {
        unr_plan_destroy(plan_h[i]);
    }

    unr_finalize();
    MPI_Finalize();
    return 0;
}