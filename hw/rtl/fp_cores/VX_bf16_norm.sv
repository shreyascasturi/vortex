

module VX_bf16_norm (
  input wire signed [15:0] exp,
  input wire [31:0] man,

  output wire signed [15:0] norm_exp,
  output wire [31:0] norm_man
);

  wire width_lzc_o;
  wire signed [7:0] width; 
  wire signed [7:0] width_diff;

  assign width[7:5] = 0'b0;

  VX_lzc #(
    .N(32),
    .MODE(1)
  ) width_lzc (
    .in_i( man >> 2 ),
    .cnt_o( width[4:0] ),
    .valid_o( width_lzc_o )
  );
  `UNUSED_VAR(width_lzc_o)

  assign width_diff = width - 7;


  wire [31:0] norm_man_rs;
  VX_bf16_rshft nm_rshft (
    .value(man),
    .n({8'b0, width_diff}),
    .shifted(norm_man_rs)
  );

  assign norm_exp = exp + {8'b0, width_diff};
  assign norm_man = (width_diff <= 0) ? man << (-width_diff) : norm_man_rs;

endmodule
