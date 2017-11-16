#ifndef _SAMPEN_NEON_H_
#define _SAMPEN_NEON_H_

#include <vector>
#include <atomic>

#include <pthread.h>
#include <semaphore.h>

void *thread_routine_wrapper(void *cookie);

class SampEnExtractor
{
public:

    typedef struct {
		float tolerance;
        float *raw_data;
        unsigned int raw_data_sz;
        float res;
    } sampen_task_t;

    typedef struct {
        unsigned int thread_num;
        SampEnExtractor *see;
        std::vector<sampen_task_t> data;
    } thread_workload_t;

    SampEnExtractor();
    ~SampEnExtractor();

    void init_thread_pool(unsigned int num_threads, int thread_sched_priority = 99);
    void cleanup_thread_pool();

    std::vector<float> extract_sampen_neon_parallel(std::vector<std::vector<float> > &data, std::vector<float> tolerances);

    static float extract_sampen_neon(const float *data, const unsigned int data_sz, float tolerance);

private:

    std::vector<pthread_t> _thread_pool;
    std::vector<thread_workload_t> _thread_cookies;

    std::atomic_ushort _th_kill;
    std::atomic_ushort _th_working;
    std::atomic_ushort _th_woken_up;

    sem_t _sem_begin;
    sem_t _sem_end;

    void _thread_routine(void *);

    friend void *thread_routine_wrapper(void *cookie);
};

#endif // _SAMPEN_NEON_H_
