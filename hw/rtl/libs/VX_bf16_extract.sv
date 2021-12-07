
// extract bits from bf16 raw floating point, unbiasing exp & normalizing 
// this will also automatically zero out subnormals
module VX_bf16_extract (
  input wire [31:0] raw_fp,

  output wire sign,
  output wire signed [15:0] exp,
  output wire [31:0] man
);

  assign sign = raw_fp[31];
  assign  exp = $signed({8'h00, raw_fp[30:23]}) - 127;
  assign  man = {25'b1, raw_fp[22:16]};

  `UNUSED_VAR(raw_fp[15:0])
endmodule
