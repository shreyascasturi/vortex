#include "bf16.h"

// This function just extracts some info about a raw bf16, including if it's normal/subnormal,
// zero, infinity, nan, etc
bf16info_t bf16_extract_info(uint32_t f);

// Extracts out the UNBIASED, signed exponent, and gets the mantissa into a properly
// sized int size for calculations
exp_man_t decode_exp_man(bf16info_t const* const b);

// Given a mantissa (mid calculation, with 2 rounding bits), a sign, and a rounding mode, will
// return the appropriately rounded mantissa
uint32_t bf16_man_round(bool sign, uint32_t man, uint32_t frm);

// Right shifts a 32-bit value by an amount, setting the 0th bit to the OR of all lost bits, and
// returns that newly shifted value
uint32_t right_shift(uint32_t val, uint32_t amnt);

// Given a sign and a rounding mode, will return the appropriate, finalized overflowed bf16 
uint32_t bf16_round_overflow(bool sign, uint32_t frm, uint32_t *fflags);

// Given all the normalized pieces of the floating point, returns a finalized, rounded bf16
uint32_t bf16_round(bool sign, int32_t exp, uint32_t man, uint32_t frm, uint32_t* fflags);

// Given all the UN-NORMALIZED pieces of a floating point, will normalize and then round them
uint32_t normalize_and_round(bool sign, int16_t exp, uint32_t man, uint32_t frm, uint32_t* fflags);


// well, here ya go
uint32_t rv_bf16_fmadd(uint32_t a, uint32_t b, uint32_t c, uint32_t frm, uint32_t* fflags) {
	bf16info_t infoA = bf16_extract_info(a),
	           infoB = bf16_extract_info(b),
			   infoC = bf16_extract_info(c);

	bf16_t bfA = infoA.val, bfB = infoB.val, bfC = infoC.val;

	// Intermediary values needed for multiplication step.
	bool signProd;
	int16_t expProd;
	uint32_t manProd;


	// first, get sign. This matters so early, bc even infinities and zeros have sign
	signProd = bfA.s ^ bfB.s;

	// CHECK FOR INF & NAN

	// if either is NaN, just propagate
	if(infoA.is_nan || infoB.is_nan){
		if(infoA.is_signalling || infoB.is_signalling){
			*fflags |= EX_INVALIDOP;
		}

		return BF16_POSQNAN; // +NaN, quiet
	}

	// if either a or b is infinity...
	if(infoA.is_inf || infoB.is_inf){
		if(infoA.is_zero || infoB.is_zero){ // Inf * 0
			// ERROR OH NO; Inf * 0 is very undefined
			*fflags |= EX_INVALIDOP;
			return BF16_POSQNAN; // +Nan, quiet
		}

		if(infoC.is_nan){ // Inf + NaN
			if(infoC.is_signalling){
				*fflags |= EX_INVALIDOP;
			}
			return BF16_POSQNAN; // +Nan, quiet
		}

		if(infoC.is_inf && bfC.s != signProd){ // Inf - Inf
			*fflags |= EX_INVALIDOP;
			return BF16_POSQNAN; // +Nan, quiet
		}

		// otherwise, just return infinity w/ the appropriate sign
		expProd = 0xFF; // infinity is all bits set exponent...
		manProd = 0;    // ...and no bits set mantissa

		return packToBF16UI(signProd, expProd, manProd);
	}

	if(infoC.is_nan){
		if(infoC.is_signalling){
			*fflags |= EX_INVALIDOP;
		}
		return c;
	}

	if(infoC.is_inf){
		return c;
	}

	// check for zeros...
	if(infoA.is_zero || infoB.is_zero){
		if(infoC.is_zero){	// If c is zero, round to positive or negative zero depending on FRM
			return packToBF16UI((frm == FRM_RDN) ? 1 : 0, 0, 0);
		}
		return c;
	}

	// Now, do the math!
	
	// First, normalize. The exponents here are unbiased
	exp_man_t aem = decode_exp_man(&infoA),
	          bem = decode_exp_man(&infoB);
	

	// Next, add exponents, subtracting out mantissa size for some reason?
	expProd = aem.exp + bem.exp - 7;

	// Next, multiply out significands. Need to reserve only two bits for rounding, so can shift
	// back one of the normalized mantissas
	manProd = (uint32_t)(aem.man >> 2) * (uint32_t)bem.man;


	// Done with the multiplication! Now, if we're adding zero, we can leave early.
	if(infoC.is_zero){
		return normalize_and_round(signProd, expProd, manProd, frm, fflags);
	}

	// From this point on, z is treated as the final value, aka z = (a*b)+c
	// ALSO, should treat all xyzProd values as CONSTANT

	// Intermediary values needed for addition step.
	bool signZ;
	int32_t expZ, expC;
	uint32_t manZ, manC;

	// Otherwise, time to begin addition.
	// First, get normalized and unbiased values for c.
	exp_man_t cem = decode_exp_man(&infoC);
	
	// Here, we are shifting the significand so there'll be enough room when adding for rounding.
	// Honestly, not quite sure why we need this.
	expC = cem.exp - 7;
	manC = cem.man << 7;
	
	// Super important, time to ALIGN the product and c, so they share an exponent.
	// We ALWAYS right shift, because we never want to lose the more significant bits on the 
	// left side. (i.e. we'd rather lose 0.0078125 than 0.5)
	if(expC < expProd){
		manC = right_shift(manC, (uint32_t)(expProd - expC));

		manZ = manProd;
		expZ = expProd;
	} else if(expC > expProd){
		manZ = right_shift(manProd, (uint32_t)(expC - expProd));
		expZ = expC;
	} else {
		manZ = manProd;
		expZ = expProd;
	}
	// At this point, expZ == expC and manC is ready to be added to manZ

	if(bfC.s == signProd){ // if they have the same sign, just add the mantissas!
		signZ = signProd;
		manZ = manZ + manC;
	} else { // otherwise, they have opposing signs...
		if(manZ == manC){ // if the mantissas are the SAME, we just return zero
			return packToBF16UI((frm == FRM_RDN) ? 1 : 0, 0, 0);
		}

		if(manZ > manC){ // if product is bigger, subtract that way
			manZ = manZ - manC;
			signZ = signProd;
		} else { // if c is bigger, subtract THAT way
			manZ = manC - manZ;
			signZ = bfC.s;
		}
	}

	return normalize_and_round(signZ, expZ, manZ, frm, fflags); 
}


bf16info_t bf16_extract_info(uint32_t f){
	bf16info_t res;
	
	res.val = { signBF16UI(f), expBF16UI(f), manBF16UI(f) };

	res.is_normal    = (res.val.exp != 0 && res.val.exp != 0xFF);
	res.is_zero      = (res.val.exp == 0 && res.val.man == 0);
	res.is_subnormal = (res.val.exp == 0 && res.val.man != 0);
	res.is_inf       = (res.val.exp == 0xFF && res.val.man == 0);
	res.is_nan       = (res.val.exp == 0xFF && res.val.man != 0);
	res.is_signalling = res.is_nan && ((res.val.man & 1) != 0);
	res.is_quiet = res.is_nan && !res.is_signalling;

	return res;
}

// Essentially, this translates the "encoded" exponent/mantissa into the actual numbers
// that should be used: unbiasing the exponent, properly handling subnormals, and addding
// the implied '1' to the normals.
exp_man_t decode_exp_man(bf16info_t const* const b){
	exp_man_t em;

	if(b->val.exp == 0){ // if subnormal...
		// the difference in widths is MANTISSA_WIDTH - #_USED_BITS + 1
		uint16_t widthDiff = 7 - ((8-1) - countLeadingZeros8(b->val.man & 0x7f));

		// normalized exponent = MIN_EXP - widthDiff
		em.exp = (int_fast16_t) -126 - widthDiff;

		// normalized mantissa = mantissa << widthDiff, and shift 2 more for round & sticky bits
		em.man = (uint_fast16_t)b->val.man << (widthDiff + 2);
	} else { // if normal...
		// normalized exponent = biased_exp - BIAS
		em.exp = (int_fast16_t) b->val.exp - 0x7f;

		// normalized mantissa = (mantissa | 0b10000000) ; aka adds implied '1' to mantissa.
		//                                               ; also shifts 2 more for round&sticky
		em.man = ((uint_fast16_t)b->val.man | (0b1 << 7)) << 2;
	}

	return em;
}

uint32_t bf16_man_round(bool sign, uint32_t man, uint32_t frm){
	if((man & 0b11) != 0){ // inexact

		switch(frm){
		case FRM_RNE:
			// if the last actual (i.e. non-rounding) bit is one, add 0b10, otherwise add 0b01
			return (man + ((man >> 2) & 1) + 1) >> 2;
		case FRM_RTZ:
			return man >> 2; // just truncate out the rounding bits
		case FRM_RDN:
			return ((sign) ? (man + 0b11) : man) >> 2;
		case FRM_RUP:
			return ((!sign) ? (man + 0b11) : man) >> 2;
		case FRM_RMM:
			return (man + 0b10) >> 2;
		}
	}
	
	return man >> 2;
}

uint32_t right_shift(uint32_t val, uint32_t amnt){
	uint32_t residue, shifted;
	if(amnt >= 32){
		residue = val;
		shifted = 0;
	} else {
		residue = val & ((1 << amnt) - 1);
		shifted = val >> amnt;
	}

	if(residue != 0) shifted |= 1;

	return shifted;
}

uint32_t bf16_round_overflow(bool sign, uint32_t frm, uint32_t *fflags){
	*fflags = *fflags | EX_OVERFLOW | EX_INEXACT;

	if((sign && frm == FRM_RUP) || (!sign && frm == FRM_RDN) || (frm == FRM_RTZ)){
		return packToBF16UI(sign, 0b11111110, 0b1111111); // round to max finite #
	}

	return packToBF16UI(sign, 0b11111111, 0b0); // appropriately signed infinity
}

// input must be normal, exponent must be unbiased
uint32_t bf16_round(bool sign, int32_t exp, uint32_t man, uint32_t frm, uint32_t* fflags){
	if(exp > 0x7F){
		return bf16_round_overflow(sign, frm, fflags);
	}

	bool roundedSign = sign;
	uint16_t roundedExp = exp;


	uint32_t roundedMan = man;
	if(exp < -126){ // if the value would be subnormal, make it such
		roundedMan = right_shift(roundedMan, (uint32_t)(-126 - exp));
	}
	
	// it's exact if the last two rounding bits are zero (and thus we dont have to round)
	bool exact = ((roundedMan & 0b11) == 0);
	roundedMan = bf16_man_round(sign, roundedMan, frm);

	if(!exact){
		*fflags |= EX_INEXACT;
		
		// if mantissa is all ones (0b1111111), rounding would cause it to
		// become 0b10000000. In this case, the mantissa will properly be 
		// truncated, but with the overflow we still need to increment the
		// exponent.
		if(roundedMan == (0b100000000)){
			roundedExp++;
			roundedMan >>= 1;
		}
	}

	// if we're subnormal or there's an underflow
	if(exp < -126){
		if(roundedMan == (1 << 7)){
			if(bf16_man_round(sign, man, frm) != (0b100000000)){
				*fflags |= EX_UNDERFLOW;
			}

			roundedExp = 1;
			roundedMan = 0;

			return packToBF16UI(roundedSign, roundedExp, roundedMan);
		}

		if(!exact){
			*fflags |= EX_UNDERFLOW;
		}

		roundedExp = 0;

		return packToBF16UI(roundedSign, roundedExp, roundedMan);
	}

	if(exp > 0x7f){
		return bf16_round_overflow(sign, frm, fflags);
	}

	

	return packToBF16UI(roundedSign, (uint16_t)(roundedExp+0x7f), (uint32_t) roundedMan);
}


uint32_t normalize_and_round(bool sign, int16_t exp, uint32_t man, uint32_t frm, uint32_t* fflags){
	// Calculate the number of bits being USED in our mantissa.
	// shift over 2 so we don't count the rounding bits in our width calc
	int32_t width = (16 - countLeadingZeros16(man>>2))-1;
	int32_t widthDiff = width - 7;


	int32_t expNorm = (int32_t)exp + widthDiff;
	
	uint32_t manNorm;

	if(widthDiff <= 0){ // if the widthDiff is negative, shift left by the positive value
		manNorm = man << ((uint32_t)(-widthDiff));
	} else { // otherwise, right shift by widthDiff
		manNorm = right_shift(man, widthDiff);
	}

	
	return bf16_round(sign, expNorm, manNorm, frm, fflags);
}


