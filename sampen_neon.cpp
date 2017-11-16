#include "sampen_neon.h"

#include <arm_neon.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

void *thread_routine_wrapper(void *cookie)
{
    SampEnExtractor::thread_workload_t *rwc = static_cast<SampEnExtractor::thread_workload_t *>(cookie);
    rwc->see->_thread_routine(cookie);
    return NULL;
}

static inline uint32_t uint32x4_check(uint32x4_t in)
{
    return in[0] & in[1] & in[2] & in[3] & 0x01;
}

void SampEnExtractor::_thread_routine(void *cookie)
{
    thread_workload_t *tmp = static_cast<thread_workload_t *>(cookie);

    printf("Thread %d started\r\n", tmp->thread_num);

    unsigned int cnt = 0;
    do {
        sem_wait(&_sem);

        ++_th_woken_up;

        ++cnt;

        ++_th_working;

        for (unsigned int i = 0; i < tmp->data.size(); ++i) {
            tmp->data[i].res = extract_sampen_neon(tmp->data[i].raw_data, tmp->data[i].raw_data_sz, _tolerance);
        }

        --_th_working;

    } while (!_th_kill);

    printf("Thread %d ran %d times\r\n", tmp->thread_num, cnt - 1);
}

SampEnExtractor::SampEnExtractor() : SampEnExtractor(0)
{
}

SampEnExtractor::SampEnExtractor(float tolerance) : _tolerance(tolerance)
{
    sem_init(&_sem, 0, 0);
}

SampEnExtractor::~SampEnExtractor()
{
    sem_destroy(&_sem);
}

void SampEnExtractor::set_tolerance(float tolerance)
{
	_tolerance = tolerance;
}

void SampEnExtractor::init_thread_pool(unsigned int num_threads, int thread_sched_priority)
{
    if (!_thread_pool.empty())
        cleanup_thread_pool();

    _thread_pool.resize(num_threads);
    _thread_cookies.resize(num_threads);

    _th_kill = 0;
    _th_working = 0;

    pthread_attr_t attr;
    memset(&attr, 0, sizeof(pthread_attr_t));

    struct sched_param sp;
    memset(&sp, 0, sizeof(sp));
    sp.sched_priority = thread_sched_priority;

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);
    pthread_attr_setschedpolicy(&attr, SCHED_FIFO);
    pthread_attr_setschedparam(&attr, &sp);

    for (unsigned int i = 0; i < num_threads; ++i) {
        _thread_cookies[i].thread_num = i;
        _thread_cookies[i].see = this;
        pthread_create(&_thread_pool[i], &attr, &thread_routine_wrapper, &_thread_cookies[i]);
    }

    pthread_attr_destroy(&attr);
}

void SampEnExtractor::cleanup_thread_pool()
{
    _th_kill = 1;

    for (unsigned int i = 0; i < _thread_pool.size(); ++i) {
        _thread_cookies[i].data.clear();
    }

    for (unsigned int i = 0; i < _thread_pool.size(); ++i) {
        sem_post(&_sem);
    }

    for (unsigned int i = 0; i < _thread_pool.size(); ++i) {
        pthread_join(_thread_pool[i], NULL);
    }

    _thread_pool.clear();
    _thread_cookies.clear();
}

std::vector<float> SampEnExtractor::extract_sampen_neon_parallel(std::vector<std::vector<float> > &data)
{
    std::vector<float> results;

    sampen_task_t workload;

    for (unsigned int i = 0; i < _thread_pool.size(); ++i) {
        _thread_cookies[i].data.clear();
    }

    workload.res = 0;
    for (unsigned int i = 0; i < data.size(); ++i) {

        workload.raw_data = data[i].data();
        workload.raw_data_sz = data[i].size();

        _thread_cookies[i % _thread_pool.size()].data.push_back(workload);
    }

    _th_woken_up = 0;

    for (int i = _thread_pool.size() - 1; i >= 0; --i) {
        sem_post(&_sem);
    }

    while (_th_woken_up < _thread_pool.size());

    while (_th_working > 0);

    for (unsigned int i = 0; i < data.size(); ++i) {
        results.push_back(_thread_cookies[i % _thread_pool.size()].data[i / _thread_pool.size()].res);
    }

    return results;
}

float SampEnExtractor::extract_sampen_neon(const float *data, const unsigned int data_sz, float tolerance)
{
    uint32_t c = 0, c1 = 0;
    uint32_t tmp;

    const float32x4_t max_err_vec = vmovq_n_f32(tolerance);
    const uint32x4_t onex4 = vmovq_n_u32(1);

    const float *data_end = data + data_sz;

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
            if (tmp > 0 && (fabs(*ni - *nj) <= tolerance))
                ++c1;
        }
    }


    return c1 > 0 ? logf((float)c / (float)c1) : 0;
}

