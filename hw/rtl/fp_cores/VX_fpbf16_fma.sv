`include "VX_fpu_define.vh"
`include "VX_bf16.vh"

module VX_fpbf16_fma #( 
    parameter TAGW = 1,
    parameter LANES = 1
) (
    input wire clk,
    input wire reset,   

    output wire ready_in,
    input wire  valid_in,

    input wire [TAGW-1:0] tag_in,
    
    input wire [`INST_FRM_BITS-1:0] frm,

    input wire  do_madd,
    input wire  do_sub,
    input wire  do_neg,

    input wire [LANES-1:0][31:0]  dataa,
    input wire [LANES-1:0][31:0]  datab,
    input wire [LANES-1:0][31:0]  datac,
    output wire [LANES-1:0][31:0] result,  

    output wire has_fflags,
    output fflags_t [LANES-1:0] fflags,

    output wire [TAGW-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);

    wire stall = ~ready_out && valid_out;
    wire enable = ~stall;

    for (genvar i = 0; i < LANES; i++) begin       
        reg [31:0] a, b, c;

        always @(*) begin
            if (do_madd) begin
                // MADD/MSUB/NMADD/NMSUB
                a = do_neg ? {~dataa[i][31], dataa[i][30:0]} : dataa[i];                    
                b = datab[i];
                c = (do_neg ^ do_sub) ? {~datac[i][31], datac[i][30:0]} : datac[i];
            end else begin
                if (do_neg) begin
                    // MUL
                    a = dataa[i];
                    b = datab[i];
                    c = 0;
                end else begin
                    // ADD/SUB
                    a = 32'h3f800000; // 1.0f
                    b = dataa[i];
                    c = do_sub ? {~datab[i][31], datab[i][30:0]} : datab[i];
                end
            end    
        end

        wire            sign_a, sign_b, sign_c;
        wire[31:0]         man_a, man_b, man_c;
        wire signed [15:0] exp_a, exp_b, exp_c;

        wire prod_sign;
        wire[15:0] prod_exp;
        wire[31:0] prod_man;

        VX_bf16_extract a_ex ( .raw_fp(a), .sign(sign_a), .exp(exp_a), .man(man_a) );
        VX_bf16_extract b_ex ( .raw_fp(b), .sign(sign_b), .exp(exp_b), .man(man_b) );
        VX_bf16_extract c_ex ( .raw_fp(c), .sign(sign_c), .exp(exp_c), .man(man_c) );


        
        assign prod_sign = sign_a ^ sign_b;
        
        // Check for inf, zero, or nan in product params
        wire found_inf_ab, found_nan_ab, found_zero_ab;
        assign found_inf_ab   = (`vx_bf16_is_inf(a)  || `vx_bf16_is_inf(b));
        assign found_nan_ab   = (`vx_bf16_is_nan(a)  || `vx_bf16_is_nan(b));
        assign found_zero_ab  = (`vx_bf16_is_zero(a) || `vx_bf16_is_zero(b));

        // Now for C
        wire found_inf_c, found_nan_c;
        assign found_inf_c  = `vx_bf16_is_inf(c);
        assign found_nan_c  = `vx_bf16_is_nan(c);
        //assign found_zero_c = `vx_bf16_is_zero(c);

        // Learned that there is no signalling NaN in bf16, so ignoring that
        wire is_special_result; // if is_special_result, then will ignore regular output and send special_result
        wire inf_ab_special; // intermediate
        wire[31:0] special_result;

        assign is_special_result = (found_nan_ab || found_inf_ab || found_nan_c || found_inf_c || found_zero_ab);
        assign inf_ab_special = (found_zero_ab || found_nan_c || (found_inf_c && sign_c != prod_sign));


                                // if A or B is nan, res=NaN 
        assign special_result = (found_nan_ab) ? {1'b0, 8'hFF, 7'b1, 16'b0} : 

                                // if A or B is Inf, and there is some other confounding factor, res=NaN. 
                                // Otherwise, if A or B is Inf, res=Inf
                                (found_inf_ab) ? ((inf_ab_special) ?  {1'b0, 8'hFF, 7'b1, 16'b0}  : 
                                                                      {1'b0, 8'hFF, 7'b0, 16'b0}) :

                                // if C is Nan or Inf, or A or B is zero, res=C. 
                                // Otherwise, no special result
                                (found_nan_c || found_inf_c || found_zero_ab) ? c : 32'b0;
        
        // Also learned that BF16 has no subnormal numbers too late, whoops
        assign prod_exp = exp_a + exp_b - 7; // don't have to unbiased bc they are already unbiased

        // shift one mantissa back 2, only need one to be shifted for rounding, otherwise we'd have 4
        assign prod_man = (man_a >> 2) * (man_b);

        // more wires for addition step
        wire z_sign;
        wire signed [15:0] z_exp;
        wire[15:0] diff_exp;
        wire[31:0] z_man;

        wire[31:0] man_c_mod, man_c_shft, prod_man_shft, prod_man_prez, man_c_prez; // all intermediaries

        assign man_c_mod = man_c << 7; // shift by mantissa to leave enough room for rounding

        assign diff_exp = (exp_c > prod_exp) ? exp_c - prod_exp : prod_exp - exp_c;

        VX_bf16_rshft man_c_shfter(
          .value(man_c_mod),
          .n(diff_exp),
          .shifted(man_c_shft)
        );

        VX_bf16_rshft prod_man_shfter(
          .value(prod_man),
          .n(diff_exp),
          .shifted(prod_man_shft)
        );
  
        // select which values to use before doing Z calculation (pre-Z)
        assign prod_man_prez = (exp_c > prod_exp) ? prod_man_shft : prod_man;
        assign man_c_prez = (exp_c > prod_exp) ? man_c_mod : man_c_shft;

                       // if the signs align, add the mantissas
        assign z_man = (sign_c == prod_sign)        ? prod_man_prez + man_c_prez :
                       // if Product man is greater, subtract such that it is positive
                       (prod_man_prez > man_c_prez) ? prod_man_prez - man_c_prez :
                       // if C man is greater, subtract such that it is positive
                       (prod_man_prez < man_c_prez) ? man_c_prez - prod_man_prez :
                       // otherwise (aka they are opposite sign but EQUAL) set to zero
                        32'b0;

                      // if C exponent is greater, use that. Otherwise, use Products
        assign z_exp = (exp_c > prod_exp) ? exp_c : prod_exp; // get the exponent to use

                        // if signs align, just choose one of them, doesn't matter
        assign z_sign = (sign_c == prod_sign) ? prod_sign :
                        // if product is greater, use product's sign
                        (prod_man_prez > man_c_prez) ? prod_sign :
                        // if C is greater, use C's sign
                        (prod_man_prez < man_c_prez) ? sign_c :
                        // otherwise, they are equal, use sign depending on rounding mode
                        (frm == `INST_FRM_RDN) ? 1'b1 : 1'b0;

        
        wire signed [15:0] norm_exp, round_exp;
        wire [31:0] norm_man, round_man;
        wire is_overflow, is_exact;

        VX_bf16_norm normalizer (
          .man(z_man),
          .exp(z_exp),
          .norm_exp(norm_exp),
          .norm_man(norm_man)
        );

        VX_bf16_round rounder (
          .exp(norm_exp),
          .man(norm_man),

          .exp_rounded(round_exp),
          .man_rounded(round_man),

          .is_overflow(is_overflow),
          .is_exact(is_exact)
        );

        //assign has_fflags = (is_overflow | is_exact);
        //assign fflags = (is_overflow) ? ;
        assign has_fflags = 0;
        assign fflags = 0;

        `UNUSED_VAR(is_overflow);
        `UNUSED_VAR(is_exact);

        wire [31:0] packed_res;
        VX_bf16_pack packer (
          .sign(z_sign),
          .exp(round_exp),
          .man(round_man),
          .fp(packed_res)
        );

        assign result[i] = (is_special_result) ? special_result : packed_res;

    end
    
    VX_shift_register #(
        .DATAW  (1 + TAGW),
        .DEPTH  (`LATENCY_FMA),
        .RESETW (1)
    ) shift_reg (
        .clk(clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({valid_in,  tag_in}),
        .data_out ({valid_out, tag_out})
    );

    assign ready_in = enable;

    `UNUSED_VAR (frm)
    assign has_fflags = 0;
    assign fflags = 0;

endmodule
