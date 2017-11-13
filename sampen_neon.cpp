#include "sampen_neon.h"

#include <string.h>
#include <pthread.h>

#include <thread>
#include <chrono>
#include <vector>
#include <mutex>
#include <iostream>

#define NUM_THREADS 3
#define NUM_CPU 4

static inline uint32_t uint32x4_check(uint32x4_t in)
{
    return in[0] & in[1] & in[2] & in[3] & 0x01;
}

float extractSampEn_neon(const float *data, float r, float sigma)
{
    uint32_t c = 0, c1 = 0;
    uint32_t tmp;
    const float max_err = r * sigma;

    const float32x4_t max_err_vec = vmovq_n_f32(max_err);
    const uint32x4_t onex4 = vmovq_n_u32(1);

    const float *data_end = data + SAMPLE_SIZE;

    for (const float *i = data; i < data_end - 8; ++i) {
        float32x4_t vec128a = vld1q_f32(i);
        for (const float *j = i + 1; j < data_end - 4; ++j) {
            float32x4_t vec128b = vld1q_f32(j);

            tmp = vminvq_u32(vandq_u32(vcleq_f32(vabdq_f32(vec128a, vec128b), max_err_vec), onex4));
            c += tmp;

            const float *ni = i + 4, *nj = j + 4;
            if (fabs(*ni - *nj) < max_err)
                ++c1;
        }
    }

    /*
    uint8_t results[SAMPLE_SIZE][SAMPLE_SIZE] = {0};

    for (unsigned int i = 0; i < SAMPLE_SIZE; ++i) {
        for (unsigned int j = 0; j < SAMPLE_SIZE; ++j) {
            results[i][j] = 0;
        }
    }

    for (unsigned int n = 0; n < 4; ++n) {
        for (unsigned int m = 0; m < 4; ++m) {
            for (unsigned int i = 0; i < nq - 1; ++i) {
                for (unsigned int j = m > n ? i : i + 1; j < nq - 1; ++j) {
                    ++results[i*4 + n][j*4 + m];
                }
            }
        }
    }

    std::ofstream out_log;
    out_log.open("log.csv", std::ios::trunc);
    for (unsigned int i = 0; i < SAMPLE_SIZE; ++i) {
        for (unsigned int j = 0; j < SAMPLE_SIZE; ++j) {
            out_log << (int)results[i][j];
    		if(j < SAMPLE_SIZE-1)
    			out_log << ",";
        }
        out_log << "\r\n";
    }
    out_log.close();
    */

    /*
    for (int n = 0; n < 4; ++n) {
        for (int m = 0; m < 4; ++m) {
            for (unsigned int i = 0; i < nq - 1; ++i) {
                //for (int i = nq - 2; i >= 0; --i) {
                base_current = data_shifts[n][i];
                base_next = data_shifts[n][i + 1];

                for (unsigned int j = (m > n ? i : i + 1); j < nq - 1; ++j) {
                    //for (int j = nq - 2; j >= (m > n ? i : i + 1); --j) {
                    compare_current = data_shifts[m][j];
                    compare_next = data_shifts[m][j + 1];
                    // abdq = absolute difference
                    // cleq = compare less than or equal (all ones if true)

                    tmp_current = vandq_u32(vcleq_f32(vabdq_f32(base_current, compare_current), max_err_vec), log_and);
                    tmp_next = vandq_u32(vcleq_f32(vabdq_f32(base_next, compare_next), max_err_vec), log_and);
                    tmp = vminvq_u32(tmp_current);

                    //tmp = uint32x4_check(vcleq_f32(vabdq_f32(base_current, compare_current), max_err_vec));

                    c += tmp;
                    c1 += tmp & vgetq_lane_u32(tmp_next, 0);

                    //c1 += tmp & static_cast<uint32_t>(fabs(base_next[0] - compare_next[0]) < max_err);

                    //                    if (uint32x4_check(vcleq_f32(vabdq_f32(base_current, data_shifts[m][j]), max_err_vec))) {
                    //                        ++c;
                    //                        if (fabs(base_next[0] - data_shifts[m][j + 1][0]) < max_err)
                    //                            ++c1;
                    //                    }

                    tmp_next = tmp_current;
                }
            }
        }
    }
    */
    return c1 > 0 ? logf((float)c / (float)c1) : 0;
}

typedef struct {
    unsigned int r;
    unsigned int sigma;
    const float *raw_data;

    float res;
} smpen_par_t;

typedef struct {
    volatile bool working;
    smpen_par_t data[10];
    unsigned int sz;
} routine_struct_t;

static pthread_t thread_pool[NUM_THREADS];
static routine_struct_t cookies[NUM_THREADS];
static volatile bool kill_threads;

static void *thread_routine(void *cookie)
{
    routine_struct_t *tmp = static_cast<routine_struct_t *>(cookie);
    unsigned int cpu_affinity;

    cpu_set_t affinity;
    sched_getaffinity(0, sizeof(cpu_set_t), &affinity);
    for (unsigned int i = 0; i < NUM_CPU; ++i) {
        if (CPU_ISSET(i, &affinity)) {
            cpu_affinity = i;
            std::cout << "Thread started on cpu " << cpu_affinity << std::endl;
        }
    }

    while (!kill_threads) {
        if (tmp->working) {
            for (unsigned int i = 0; i < tmp->sz; ++i) {
                tmp->data[i].res = extractSampEn_neon(tmp->data[i].raw_data, tmp->data[i].r, tmp->data[i].sigma);
            }
            tmp->working = false;
        }
    }
    return NULL;
}

void init_neon_parallel()
{
    pthread_attr_t attr = {0};
    cpu_set_t affinity;

    kill_threads = false;

    struct sched_param sp;
    memset(&sp, 0, sizeof(sp));
    sp.sched_priority = 99;

    pthread_attr_init(&attr);

    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);
    pthread_attr_setschedpolicy(&attr, SCHED_FIFO);
    pthread_attr_setschedparam(&attr, &sp);
    for (unsigned int i = 0; i < NUM_THREADS; ++i) {
        CPU_ZERO(&affinity);
        CPU_SET(i % (NUM_CPU - 1), &affinity);
        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &affinity);
        pthread_create(&thread_pool[i], &attr, &thread_routine, &cookies[i]);
    }

    pthread_attr_destroy(&attr);

    CPU_SET(NUM_CPU - 1, &affinity);
    sched_setaffinity(0, sizeof(cpu_set_t), &affinity);
}

void cleanup_neon_parallel()
{
    kill_threads = true;

    for (unsigned int i = 0; i < NUM_THREADS; ++i) {
        cookies[i].working = false;
    }

    for (unsigned int i = 0; i < NUM_THREADS; ++i) {
        pthread_join(thread_pool[i], NULL);
    }
}

std::vector<float> extractSampEn_neon_parallel(std::vector<std::vector<float> > &data, float r, float sigma)
{
    std::vector<float> ret_value;

    smpen_par_t routine_data;
    routine_data.r = r;
    routine_data.sigma = sigma;

    for (unsigned int i = 0; i < NUM_THREADS; ++i) {
        cookies[i].sz = 0;
    }

    for (unsigned int i = 0; i < data.size(); ++i) {
        routine_data.raw_data = data[i].data();
        cookies[i % NUM_THREADS].data[i / NUM_THREADS] = routine_data;
        ++cookies[i % NUM_THREADS].sz;
    }

    for (unsigned int i = 0; i < NUM_THREADS; ++i) {
        cookies[i].working = true;
    }

    bool tmp;
    do {
        tmp = false;
        for (unsigned int i = 0; i < NUM_THREADS; ++i) {
            tmp = tmp || cookies[i].working;
        }
    } while (tmp);

    for (unsigned int i = 0; i < data.size(); ++i) {
        ret_value.push_back(cookies[i % NUM_THREADS].data[i / NUM_THREADS].res);
    }

    return ret_value;
}

