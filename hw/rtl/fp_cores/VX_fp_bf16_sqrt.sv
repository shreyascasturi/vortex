`include "VX_fpu_define.vh"
// Shreyas Casturi
// BF16 implementation of square root floating point operation
// Adapted from rust library of floating point operations
// bf16 is laid out as so:
// |[15]|[14:7]| [6:0]    |
// |    |      |          |
// |sign| exp  | mantissa |
// we consider the top 16 bits of a 32 bit number. So, [31, 16]. 
// We do not treat [15, 0].


// taken boilerplate from VX_fp_sqrt.sv
module VX_fp_bf16_sqrt #( 
    parameter TAGW = 1,
    parameter LANES = 1
) (
    input wire clk,
    input wire reset,   

    output wire ready_in,
    input wire  valid_in,

    input wire [TAGW-1:0] tag_in,
    
    input wire [`INST_FRM_BITS-1:0] frm,

    input wire [LANES-1:0][31:0]  dataa,
    // modify result to be 16 bits wide
    output wire [LANES-1:0][15:0] result,  

    output wire has_fflags,
    output fflags_t [LANES-1:0] fflags,

    output wire [TAGW-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
); 

    // this is used to create a vector of zeroes to compare against
    localparam CONST_32 = 32;
    localparam CONST_16 = 16;


// also boilerplate from VX_fp_sqrt.sv
    wire stall = ~ready_out && valid_out;
    wire enable = ~stall;
    wire [CONST_32-1:0] zero_constant_32 = {CONST_32{1'b0}};
    wire [CONST_16-1:0] zero_constant_16 = {CONST_16{1'b0}};
 
    // for each lane, do a sqrt on the given number.
    // we always work on the upper 16 bits
    for (genvar i = 0; i < LANES; i++) begin
       wire[31:0] fp_32_number = dataa[i]; 
       
       wire[15:0] upper_16_bits = fp_32_number[31:16];
       
       // to be clear, rust library uses a "get_normalized_significand". 
       // not sure if we need to
       // worry about this
       wire[7:0] exponent = upper_16_bits[14:7]; 

       wire[6:0] mantissa = upper_16_bits[6:0];

       wire sign = upper_16_bits[15];

       // i don't know how to check correctly
       // check if exponent is all ones for NaN
       // exponent must be all 1s ,mantissa non-zero
       if (exponent == 255 && mantissa != 0) begin
           // if (signaling) {
               // set divide by 0
               // return nan quiet
           //}
           result[i] = 0; // not sure how to assign nan quiets
       end
           

        // if zero
       if (upper_16_bits == zero_constant_16) begin
           result[i] = zero_constant_16;
       end
           
       
       
       // if negative
       if (sign == 1) begin
           // return nan quiet
       end
           
       
       // if infinity, return infinity
       if (exponent == 255) begin
           result[i] = upper_16_bits;
       end
           


        // if exponent divisible by 2
       if (exponent % 2 == 0) begin
           exponent--;
           mantissa = mantissa << 1;
       end
           
       
       // from rust library: result = desc::holder::one << desc::significand_width + 2
       // significand width here is 7, so 7 + 2 is 9.
       wire[9:0] temp_result = 1 << (9);

       // from rust library: halfbit = desc:holder::one << desc::significand_width.
       // so this would be 8 bits wide as sig_width = 7
       wire[7:0] half_bit = 1 << (7);

        // not sure how this works out -- mantissa is 7 bits, temp_result is 8 bits...
       wire[15:0] half_sig_minus_result = mantissa - temp_result;

        // while half bit does not equal 0
       while (half_bit != 0) begin
           // if this "result" is greater than some addition
           // decrement, add to temp result.
           if (half_sig_minus_result >= (temp_result + half_bit)) begin
               half_sig_minus_result = half_sig_minus_result - (temp_result + half_bit);
               temp_result = temp_result + (half_bit << 1);
           end
           half_bit = half_bit >> 1;
           half_sig_minus_result = half_sig_minus_result << 1;
       end

       if (half_sig_minus_result != zero_constant_16) begin
           temp_result = temp_result | 1; // not sure if this is all 1s, or just 0000....01.
       end

        // do rounding somewhere (return rounding(exponent/2, result))
        result[i] = temp_result;
       
    end
endmodule