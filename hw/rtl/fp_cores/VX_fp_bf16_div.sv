`include "VX_fpu_define.vh"
// Shreyas Casturi
// BF16 implementation of division floating point operation
// Adapted from rust library of floating point operations
// bf16 is laid out as so:
// |[15]|[14:7]| [6:0]    |
// |    |      |          |
// |sign| exp  | mantissa |
// we consider the top 16 bits of a 32 bit number. So, [31, 16]. 
// We do not treat [15, 0].

module VX_fp_div #( 
    parameter TAGW = 1,
    parameter LANES = 1
) (
    input wire clk,
    input wire reset,   

    output wire ready_in,
    input wire  valid_in,

    input wire [TAGW-1:0] tag_in,

    input wire [`INST_FRM_BITS-1:0] frm,
    
    // for now we'll assume we work on top 16 bits
    input wire [LANES-1:0][31:0]  dataa,
    input wire [LANES-1:0][31:0]  datab,
    output wire [LANES-1:0][15:0] result,  

    output wire has_fflags,
    output fflags_t [LANES-1:0] fflags,

    output wire [TAGW-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);    
    wire stall = ~ready_out && valid_out;
    wire enable = ~stall;

    // do for every lane, a division
    // we delineate as fp_32_number_A (the first number)
    // fp_32_number_B (the second number)
    for (genvar i = 0; i < LANES; i++) begin
       // get first fp number
       wire[31:0] fp_32_number_A = dataa[i];        
       wire[15:0] upper_16_bits_A = fp_32_number_A[31:16];
       
       // to be clear, rust library uses a "get_normalized_significand". 
       // not sure if we need to
       // worry about this for both numbers
       wire[7:0] exponent_A = upper_16_bits_A[14:7]; 

       wire[6:0] mantissa_A = upper_16_bits_A[6:0];

       wire sign_A = upper_16_bits_A[15];

       // get second fp number
       wire[31:0] fp_32_number_B = datab[i];
       wire[15:0] upper_16_bits_B = fp_32_number_B[31:16];

       wire[7:0] exponent_B = upper_16_bits_B[14:7]; 

       wire[6:0] mantissa_B = upper_16_bits_B[6:0];

       wire sign_B = upper_16_bits_B[15];

        // check if NAN for either number
        if ((exponent_A == 255 && mantissa_A != 0) || (exponent_B == 255 && mantissa_B != 0)) begin
            // codebase has: Self::propagate_nan(a, b), for both numbers.
            // Not sure what that means
        end

        wire sign_xor = sign_A ^ sign_B; // don't know why we need this

        // check infinity
        if (exponent_A == 255) begin
            if (exponent_B == 255) begin
                // return quiet NAN
            end
            // return infinity
        end

        wire[0:7] quotient_exponent = exponent_A - exponent_B;

        // check mantissa... "normalize" mantissa?
        if (mantissa_A < mantissa_B) begin
            mantissa_A = mantissa_A << 1;
            quotient_exponent = quotient_exponent - 1;
        end

        wire[9:0] quotient = 1 << (9); // significand/mantissa width is 7
        wire[8:0] bit_val = 1 << (8); // significand/mantissa width is 7
        wire[7:0] remainder_over_bit = (mantissa_A - mantissa_B) << 1;

        // while bit_val (which is 1 * 2^8) is not equal to 1
        while (bit_val != 1) begin
            if (remainder_over_bit >= mantissa_B) begin
                remainder_over_bit = remainder_over_bit - mantissa_B;
                quotient = quotient + bit_val;
            end

            bit_val = bit_val >> 1;
            remainder_over_bit = remainder_over_bit << 1;
        end
        // 0 = Desc::Holder::zero()
        if (remainder_over_bit != 0) begin
            quotient = quotient | 1; // 1 = Desc::Holder::one()
        end

        // do rounding: Self::round(sign, quotient_exponent, quotient)
        result[i] = quotient; // not sure what actually goes here, but it has to be something
    end
endmodule