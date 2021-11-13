#ifndef BF16_H
#define BF16_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

#define signBF16UI(a) ((bool)((uint32_t)(a) >> 15) & 1)
#define expBF16UI(a)  ((int_fast8_t) ((a)>>7) & 0xFF)
#define manBF16UI(a)  ((int_fast8_t) (a) & 0x7F)
#define packToBF16(sign, exp, man) (((uint32_t)(sign) << 15) + ((uint32_t)(exp)<<7) + (man))

// NaN is any sign, all ones for exponent, and non-zero mantissa
#define isNaNBF16UI(a) (((~(a) & 0x7F80) == 0) && ((a) & 0x7F))

// Inf is any sign, all ones for exponent, and mantissa fully zeroed
#define isInfBF16UI(a) (((~(a) & 0x7F80) == 0) && !((a) & 0x7F))


#ifdef __cplusplus
}
#endif

#endif
