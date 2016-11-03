#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <algorithm>
#include <mutex>
#include <condition_variable>
#include <future>

#include <time.h>
#include <math.h>
#include <string.h>

#include <sys/resource.h>
#include <sched.h>

#define SAMPLE_SIZE 128

#include "sampen.h"
#include "sampen_neon.h"

#define NUM_RUNS 10

/*
static void thread_func(unsigned int n, float r, float sigma)
{
    float32x4_t base_current, base_next;
    const float max_err = r * sigma;
    const float32x4_t max_err_vec = {max_err, max_err, max_err, max_err};
    const unsigned int nq = SAMPLE_SIZE / 4;
    std::unique_lock<std::mutex> lock(mtx, std::defer_lock);

    while (!kill_threads) {
        lock.lock();
        if (cond_var.wait_for(lock, std::chrono::seconds(1)) == std::cv_status::no_timeout) {
            lock.unlock();
            for (unsigned int i = 0; i < nq - 2; ++i) {
                base_current = data_shifts[n][i];
                base_next = data_shifts[n][i + 1];
                for (unsigned int m = n; m < 4; ++m) {
                    for (unsigned int j = m > n ? i : i + 1; j < nq - 1; ++j) {
                        // abdq = absolute difference
                        // cleq = compare less or equal (all ones if true)
                        // shrq = shift by finite amount to the right
                        // addq = add
                        if (uint32x4_check_gt(vcleq_f32(vabdq_f32(base_current, data_shifts[m][j]), max_err_vec))) {
                            ++c[n];
                            if (fabs(base_next[0] - data_shifts[m][j + 1][0]) < max_err)
                                ++c1[n];
                        }
                    }
                }
            }
            sem_tracker[n] = 0;
        } else
            lock.unlock();
    }
}*/

static inline double elapsed_ms(struct timespec &start, struct timespec &end)
{
    return (end.tv_sec * 1000.0 + end.tv_nsec / 1000000.0) - (start.tv_sec * 1000.0 + start.tv_nsec / 1000000.0);
}

#define FILE_COLS 9
static std::vector<float> data[FILE_COLS];

int main(int, char **)
{
    if(setpriority(PRIO_PROCESS, 0, -20) != 0)
    	std::cerr << "Could not set process priority" << std::endl;

    struct sched_param sp;
    memset(&sp, 0, sizeof(sp));
    sp.sched_priority = 90;
    if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0)
        std::cerr << "Could not set process scheduling policy" << std::endl;

    std::ifstream data_stream("learningData.txt", std::ios::in);

    if (!data_stream) {
        std::cerr << "Open failed" << std::endl;
        return -1;
    }

    float tmp;
    while (!data_stream.eof()) {
        for (unsigned int i = 0; i < FILE_COLS; ++i) {
            data_stream >> tmp;
            data[i].push_back(tmp);
        }
    }

    data_stream.close();
    std::cout << "Loading complete" << std::endl << std::endl;

    unsigned int wnd_cnt = 0;
    struct timespec start = {0}, end = {0};
    double time_neon = 0, time_normal = 0, time_neon_threaded = 0, time_normal_threaded = 0;


    float res_normal[FILE_COLS], res_neon[FILE_COLS];
    std::future<float> futures[FILE_COLS];

    float r = 0.2, sigma = 255;
/*
    wnd_cnt = 0;
    clock_gettime(CLOCK_REALTIME, &start);
    for (unsigned int num_runs = 0; num_runs < NUM_RUNS; ++num_runs) {
        for (unsigned int i = 0; i * 64 < data[0].size() - 1; ++i) {
            ++wnd_cnt;
            for (unsigned int chan = 1; chan < FILE_COLS; ++chan) {
                res_normal[chan] = extractSampEn(data[chan].data() + i * 64, 4, r, SAMPLE_SIZE, sigma);
            }
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    time_normal = elapsed_ms(start, end);

    std::cout << "----> Not threaded" << std::endl;
    std::cout << "Average time : " << time_normal / wnd_cnt << "ms" << std::endl;

    wnd_cnt = 0;
    clock_gettime(CLOCK_REALTIME, &start);
    for (unsigned int num_runs = 0; num_runs < NUM_RUNS; ++num_runs) {
        for (unsigned int i = 0; i * 64 < data[0].size() - 1; ++i) {
            ++wnd_cnt;
            for (unsigned int chan = 1; chan < FILE_COLS; ++chan) {
                res_neon[chan] = extractSampEn_neon(data[chan].data() + i * 64, r, sigma);
            }
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    time_neon = elapsed_ms(start, end);

    std::cout << "Average time (neon) : " << time_neon / wnd_cnt << "ms" << std::endl;
    std::cout << "Average performance gain : " << (100 * (time_normal - time_neon) / time_normal) << "%" << std::endl << std::endl;

    wnd_cnt = 0;
    clock_gettime(CLOCK_REALTIME, &start);
    for (unsigned int num_runs = 0; num_runs < NUM_RUNS; ++num_runs) {
        for (unsigned int i = 0; i * 64 < data[0].size() - 1; ++i) {
            ++wnd_cnt;
            for (unsigned int chan = 1; chan < FILE_COLS; ++chan) {
                futures[chan] = std::async(std::launch::async, &extractSampEn, data[chan].data() + i * 64, 4, r, SAMPLE_SIZE, sigma);
            }
            for (unsigned int chan = 1; chan < FILE_COLS; ++chan) {
                res_normal[chan] = futures[chan].get();
            }
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    time_normal_threaded = elapsed_ms(start, end);

    std::cout << "----> Threaded" << std::endl;
    std::cout << "Average time : " << time_normal_threaded / wnd_cnt << "ms" << std::endl;

    wnd_cnt = 0;
    clock_gettime(CLOCK_REALTIME, &start);
    for (unsigned int num_runs = 0; num_runs < NUM_RUNS; ++num_runs) {
        for (unsigned int i = 0; i * 64 < data[0].size() - 1; ++i) {
            ++wnd_cnt;
            for (unsigned int chan = 1; chan < FILE_COLS; ++chan) {
                futures[chan] = std::async(std::launch::async, &extractSampEn_neon, data[chan].data() + i * 64, r, sigma);
            }
            for (unsigned int chan = 1; chan < FILE_COLS; ++chan) {
                res_neon[chan] = futures[chan].get();
            }
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    time_neon_threaded = elapsed_ms(start, end);

    std::cout << "Average time (neon) : " << time_neon_threaded / wnd_cnt << "ms" << std::endl;
    std::cout << "Average performance gain : " << (100 * (time_normal_threaded - time_neon_threaded) / time_normal_threaded) << "%" << std::endl << std::endl;
	*/
	
	
	init_neon_parallel();
	std::vector<std::vector<float> > vec_data;
	for(unsigned int i = 1; i < FILE_COLS; ++i)
	{
		vec_data.push_back(data[i]);
	}
	
    wnd_cnt = 0;
    clock_gettime(CLOCK_REALTIME, &start);
    for (unsigned int num_runs = 0; num_runs < NUM_RUNS; ++num_runs) {
        for (unsigned int i = 0; i * 64 < data[0].size() - 1; ++i) {
            ++wnd_cnt;

			extractSampEn_neon_parallel(vec_data, r, sigma);
        }
    }
    clock_gettime(CLOCK_REALTIME, &end);
    time_neon_threaded = elapsed_ms(start, end);
	cleanup_neon_parallel();

    std::cout << "----> Manually threaded" << std::endl;
    std::cout << "Average time (neon) : " << time_neon_threaded / wnd_cnt << "ms" << std::endl;
}
