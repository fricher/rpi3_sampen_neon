#include "sampen_neon.h"

#include <string.h>
#include <pthread.h>
#include <semaphore.h>

#include <thread>
#include <chrono>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <atomic>

#define NUM_CPU 4
#define NUM_THREADS 4

static inline uint32_t uint32x4_check(uint32x4_t in)
{
    return in[0] & in[1] & in[2] & in[3] & 0x01;
}

float extractSampEn_neon(const float *data, float r, float sigma, unsigned int sample_size)
{		
    uint32_t c = 0, c1 = 0;
    uint32_t tmp;
    const float max_err = r * sigma;

    const float32x4_t max_err_vec = vmovq_n_f32(max_err);
    const uint32x4_t onex4 = vmovq_n_u32(1);

    const float *data_end = data + sample_size;

    for (const float *i = data; i < data_end - 4; ++i) {
        float32x4_t vec128a = vld1q_f32(i);
        for (const float *j = i + 1; j < data_end - 4; ++j) {
            float32x4_t vec128b = vld1q_f32(j);

#ifdef __aarch64__
            tmp = vminvq_u32(vandq_u32(vcleq_f32(vabdq_f32(vec128a, vec128b), max_err_vec), onex4));
#else
            tmp = uint32x4_check(vandq_u32(vcleq_f32(vabdq_f32(vec128a, vec128b), max_err_vec), onex4));
#endif

            c += tmp;

            const float *ni = i + 4, *nj = j + 4;
            if (tmp > 0 && (fabs(*ni - *nj) <= max_err))
                ++c1;
        }
    }
    

    return c1 > 0 ? logf((float)c / (float)c1) : 0;
}

typedef struct {
    float r;
    float sigma;
    float *raw_data;
	unsigned int raw_data_sz;

    float res;
} smpen_par_t;

typedef struct {
    unsigned int num;
    smpen_par_t data[10];
    unsigned int sz;
} routine_struct_t;

static pthread_t thread_pool[NUM_THREADS];
static routine_struct_t cookies[NUM_THREADS];

std::atomic_ushort th_kill;
std::atomic_ushort th_working;
std::atomic_ushort th_woken_up;

sem_t sem;

static void *thread_routine(void *cookie)
{
    routine_struct_t *tmp = static_cast<routine_struct_t *>(cookie);

    printf("Thread %d started\r\n", tmp->num);

    unsigned int cnt = 0;
    do {
        sem_wait(&sem);

        ++th_woken_up;

        ++cnt;

        ++th_working;
		
			//printf("%d | %f %f %f %f %f\r\n",tmp->num, tmp->data[0].raw_data[tmp->data[0].raw_data_sz-5],tmp->data[0].raw_data[tmp->data[0].raw_data_sz-4],tmp->data[0].raw_data[tmp->data[0].raw_data_sz-3],tmp->data[0].raw_data[tmp->data[0].raw_data_sz-2],tmp->data[0].raw_data[tmp->data[0].raw_data_sz-1]);

        for (unsigned int i = 0; i < tmp->sz; ++i) {
            tmp->data[i].res = extractSampEn_neon(tmp->data[i].raw_data, tmp->data[i].r, tmp->data[i].sigma, tmp->data[i].raw_data_sz);
			//printf("res=%f, %d\r\n",tmp->data[i].res, tmp->data[i].raw_data_sz);
        }

        --th_working;

    } while (!th_kill);

    printf("thread %d ran %d times\r\n", tmp->num, cnt - 1);

    return NULL;
}

void init_neon_parallel()
{
    pthread_attr_t attr = {0};

    th_kill = 0;

    sem_init(&sem, 0, 0);
    th_working = 0;

    struct sched_param sp;
    memset(&sp, 0, sizeof(sp));
    sp.sched_priority = 99;

    pthread_attr_init(&attr);

    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);
    pthread_attr_setschedpolicy(&attr, SCHED_FIFO);
    pthread_attr_setschedparam(&attr, &sp);
    for (unsigned int i = 0; i < NUM_THREADS; ++i) {
        cookies[i].num = i;
        pthread_create(&thread_pool[i], &attr, &thread_routine, &cookies[i]);
    }

    pthread_attr_destroy(&attr);
}

void cleanup_neon_parallel()
{
    th_kill = 1;

    for (unsigned int i = 0; i < NUM_THREADS; ++i) {
        cookies[i].sz = 0;
    }

    for (unsigned int i = 0; i < NUM_THREADS; ++i) {
        sem_post(&sem);
    }

    for (unsigned int i = 0; i < NUM_THREADS; ++i) {
        pthread_join(thread_pool[i], NULL);
    }

    sem_destroy(&sem);
}

std::vector<float> extractSampEn_neon_parallel(std::vector<std::vector<float> > &data, float r, float sigma, unsigned int sample_size)
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
		routine_data.raw_data_sz = data[i].size();
        cookies[i % NUM_THREADS].data[i / NUM_THREADS] = routine_data;
        ++cookies[i % NUM_THREADS].sz;
    }

    th_woken_up = 0;

    for (int i = NUM_THREADS - 1; i >= 0; --i) {
        sem_post(&sem);
    }

    while (th_woken_up < NUM_THREADS);

    while (th_working > 0);

    for (unsigned int i = 0; i < data.size(); ++i) {
        ret_value.push_back(cookies[i % NUM_THREADS].data[i / NUM_THREADS].res);
    }

    return ret_value;
}

