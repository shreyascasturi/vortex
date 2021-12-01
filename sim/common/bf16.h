#ifndef BF16_H
#define BF16_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

#define UIBF16SHFT(a) ((a) >> 16)
#define UIBF16UNSHFT(a) ((a) << 16)

#define signBF16UI(a) ((bool)(((uint32_t)(UIBF16SHFT(a)) >> 15) & 1))
#define expBF16UI(a)  ((uint_fast8_t) (((UIBF16SHFT(a))>>7) & 0xFF))
#define manBF16UI(a)  ((uint_fast8_t) ((UIBF16SHFT(a)) & 0x7F))
#define packToBF16UI(sign, exp, man) UIBF16UNSHFT(((uint32_t)((sign)&0x1) << 15) + ((uint32_t)((exp)&0xFF)<<7) + (uint32_t)((man)&0x7F))

// NaN is any sign, all ones for exponent, and non-zero mantissa
#define isNaNBF16UI(a) (((~(UIBF16SHFT(a)) & 0x7F80) == 0) && ((UIBF16SHFT(a)) & 0x7F))

// Inf is any sign, all ones for exponent, and mantissa fully zeroed
#define isInfBF16UI(a) (((~(UIBF16SHFT(a)) & 0x7F80) == 0) && !((UIBF16SHFT(a)) & 0x7F))

uint32_t rv_bf16_fmadd(uint32_t a, uint32_t b, uint32_t c, uint32_t frm, uint32_t* fflags);

typedef struct _bf16_t {
	bool s;      // sign
	uint_fast8_t exp; // exponent
	uint_fast8_t man; // mantissa / significand
} bf16_t;


typedef struct _exp_n_man_t {
	int_fast16_t exp;
	uint_fast32_t man;
} exp_man_t;

typedef struct _bf16info_t {
	bool is_normal;
	bool is_subnormal;
	bool is_zero;
	bool is_inf;
	bool is_nan;
	bool is_signalling;
	bool is_quiet;

	bf16_t val;
} bf16info_t;


// Directly taken from softfloat. This is, by far, the most hilarious bit of code I've seen for this
// project. I love it and I'm stealing it.
/*const uint_least8_t softfloat_countLeadingZeros8[256] = {*/
const uint_least8_t arr_countLeadingZeros8[256] = {
    8, 7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

#define countLeadingZeros8(n) arr_countLeadingZeros8[n]

static uint_fast8_t countLeadingZeros16(uint16_t n){
	if(n <= 0xFF){ // if smaller than 0xFFFF
		return 8+countLeadingZeros8(n & 0xFF);
	} 
	
	return countLeadingZeros8((n >> 8) & 0xFF);
}

// FLAGS

#define FRM_RNE 0b000  // round to nearest, ties even
#define FRM_RTZ 0b001  // round towards zero, aka truncate
#define FRM_RDN 0b010  // round down (towards -inf)
#define FRM_RUP 0b011  // round up (towards +inf)
#define FRM_RMM 0b100  // round to nearest, ties to max mag

#define EX_INEXACT   0b00000
#define EX_UNDERFLOW 0b00010
#define EX_OVERFLOW  0b00100
#define EX_DIVBY0    0b01000
#define EX_INVALIDOP 0b10000

// CONSTANTS
#define BF16_POSQNAN 0x7f81


// DEBUGGING
#define BB "%c%c%c%c_%c%c%c%c"
#define B2B(b) \
	((b) & 0x80 ? '1' : '0'), \
	((b) & 0x40 ? '1' : '0'), \
	((b) & 0x20 ? '1' : '0'), \
	((b) & 0x10 ? '1' : '0'), \
	((b) & 0x08 ? '1' : '0'), \
	((b) & 0x04 ? '1' : '0'), \
	((b) & 0x02 ? '1' : '0'), \
	((b) & 0x01 ? '1' : '0')

#ifdef __cplusplus
}
#endif

#endif 
