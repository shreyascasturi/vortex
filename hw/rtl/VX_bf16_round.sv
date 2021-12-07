// note: bfloat16 always rounds to nearest even (aka RNE)


module VX_bf16_round (
  input wire signed[15:0] exp,
  input wire       [31:0] man,

  output wire      [15:0] exp_rounded,
  output wire      [31:0] man_rounded,

  output wire             is_overflow,
  output wire             is_exact
);


  wire [31:0] man_intermed;


  assign is_exact = ((man & 32'b11) == 0); // if round & sticky are zero, we have an exact answer

  // if not exact, perform RNE
  assign man_intermed = (!is_exact) ? (man + ((man >> 2) & 1) + 1) >> 2 : man;

  // if, in rounding, mantissa overflowed, shift over one and increment exponent
  assign man_rounded = (!is_exact && (man_intermed == 32'b10000000)) ? man_intermed >> 1 : man_intermed;
  assign exp_rounded = (!is_exact && (man_intermed == 32'b10000000)) ? exp + 1 : exp;

  assign is_overflow = (exp_rounded > 16'h7f);
endmodule
