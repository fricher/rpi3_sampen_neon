#include "sampen.h"

#include <math.h>

float extractSampEn(const float *data, unsigned int m, float r, unsigned int N, float sigma)
{
    int Cm = 0, Cm1 = 0;
    float err = 0.0;
    err = sigma * r;
    const unsigned int testI = N - (m + 1) + 1;

    for (unsigned int i = 0; i < testI; ++i) {
        for (unsigned int j = i + 1; j < testI; ++j) {
            bool eq = true;
            //m - length series
            for (unsigned int k = 0; k < m; ++k) {
                if (fabs(data[i + k] - data[j + k]) > err) {
                    eq = false;
                    break;
                }
            }
            if (eq) ++Cm;

            //m+1 - length series
            int k = m;
            if (eq && fabs(data[i + k] - data[j + k]) <= err)
                ++Cm1;
        }
    }

    //	std::cout << " normal: " << Cm << " " << Cm1 << " " << std::endl;

    if (Cm > 0 && Cm1 > 0)
        return logf((float)Cm / (float)Cm1);
    else
        return 0.0;
}
