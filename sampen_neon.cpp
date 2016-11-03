#include "sampen_neon.h"

#include <string.h>
#include <pthread.h>

#include <thread>
#include <chrono>
#include <vector>
#include <mutex>
#include <iostream>
#include <condition_variable>

#define NUM_THREADS 6
#define NUM_CPU 4

static inline uint32_t uint32x4_check(uint32x4_t in)
{
    return in[0] & in[1] & in[2] & in[3] & 0x01;
}

float extractSampEn_neon(const float *data, float r, float sigma)
{
    unsigned int c = 0, c1 = 0;

    float32x4_t data_shifts[4][SAMPLE_SIZE / 4];

    const float max_err = r * sigma;
    const unsigned int nq = SAMPLE_SIZE / 4;

    memcpy(data_shifts[0], &data[0], SAMPLE_SIZE * sizeof(float));
    memcpy(data_shifts[1], &data[1], (SAMPLE_SIZE - 1)*sizeof(float));
    memcpy(data_shifts[2], &data[2], (SAMPLE_SIZE - 2)*sizeof(float));
    memcpy(data_shifts[3], &data[3], (SAMPLE_SIZE - 3)*sizeof(float));

    float32x4_t base_current, base_next, compare_current, compare_next;
    const float32x4_t max_err_vec = {max_err, max_err, max_err, max_err};
    unsigned int starting_j;
	uint32_t tmp;

    for (unsigned int n = 0; n < 4; ++n) {
        for (unsigned int m = n; m < 4; ++m) {
            for (unsigned int i = 0; i < nq - 2; ++i) {
                base_current = data_shifts[n][i];
                base_next = data_shifts[n][i + 1];
                starting_j = m > n ? i : i + 1;
                for (unsigned int j = starting_j; j < nq - 1; ++j) {
                    compare_current = data_shifts[m][j];
                    compare_next = data_shifts[m][j + 1];
                    // abdq = absolute difference
                    // cleq = compare less than or equal (all ones if true)
                    tmp = uint32x4_check(vcleq_f32(vabdq_f32(base_current, compare_current), max_err_vec));
                    c += tmp;
                    c1 += tmp & static_cast<uint32_t>(fabs(base_next[0] - compare_next[0]) < max_err);
                    //                    if (uint32x4_check(vcleq_f32(vabdq_f32(base_current, data_shifts[m][j]), max_err_vec))) {
                    //                        ++c;
                    //                        if (fabs(base_next[0] - data_shifts[m][j + 1][0]) < max_err)
                    //                            ++c1;
                    //                    }
                }
            }
        }
    }

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
static std::condition_variable cond_var;
static std::mutex mtx;



static void *thread_routine(void *cookie)
{
    routine_struct_t *tmp = static_cast<routine_struct_t *>(cookie);
    struct timespec time;

	cpu_set_t affinity;
	sched_getaffinity(0,sizeof(cpu_set_t),&affinity);
	std::cout << "Thread started on cpu " << CPU_COUNT(&affinity) << std::endl;
	
    while (!kill_threads) {
        /*std::unique_lock<std::mutex> lock(mtx);
        if (cond_var.wait_for(lock, std::chrono::seconds(1)) == std::cv_status::no_timeout) {
            lock.unlock();*/
        if (tmp->working) {
            /*for (unsigned int i = 0; i < tmp->sz; ++i) {
                tmp->data[i].res = extractSampEn_neon(tmp->data[i].raw_data, tmp->data[i].r, tmp->data[i].sigma);
            }*/
            tmp->working = false;
        }
        //std::this_thread::sleep_for(std::chrono::microseconds(100));
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
        CPU_SET(i % (NUM_CPU-1), &affinity);
        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &affinity);
        pthread_create(&thread_pool[i], &attr, &thread_routine, &cookies[i]);
    }

    pthread_attr_destroy(&attr);
	
	CPU_SET(NUM_CPU-1,&affinity);
	sched_setaffinity(0, sizeof(cpu_set_t), &affinity);
}

void cleanup_neon_parallel()
{
    kill_threads = true;

    for (unsigned int i = 0; i < NUM_THREADS; ++i) {
        pthread_join(thread_pool[i], NULL);
    }
}

std::vector<float> extractSampEn_neon_parallel(std::vector<std::vector<float> > data, float r, float sigma)
{
    std::vector<float> ret_value;

    {
        std::lock_guard<std::mutex> lock(mtx);
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
    }
    //cond_var.notify_all();

    while (true) {
        bool tmp = false;
        for (unsigned int i = 0; i < NUM_THREADS; ++i) {
            tmp = tmp || cookies[i].working;
            //std::cout << cookies[i].working << " ";
        }
        //std::cout << tmp << std::endl;
        if (!tmp) {
            break;
        } /*else {
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }*/
    }

    for (unsigned int i = 0; i < data.size(); ++i) {
        ret_value.push_back(cookies[i % NUM_THREADS].data[i / NUM_THREADS].res);
    }

    return ret_value;
}
