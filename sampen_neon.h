#ifndef _SAMPEN_NEON_H_
#define _SAMPEN_NEON_H_

#include <arm_neon.h>
#include <math.h>

#include <vector>

#ifndef SAMPLE_SIZE
#define SAMPLE_SIZE 128
#endif

void init_neon_parallel();
void cleanup_neon_parallel();

float extractSampEn_neon(const float *data, float r, float sigma);
std::vector<float> extractSampEn_neon_parallel(std::vector<std::vector<float> > data, float r, float sigma);

#endif // _SAMPEN_NEON_H_
