


// pack a full fp back together
module VX_bf16_pack (
  input wire sign,
  input wire signed [15:0] exp,
  input wire [31:0] man,

  output wire [31:0] fp
);

  `UNUSED_VAR(exp[15:8])
  `UNUSED_VAR(man[31:7])

  //assign fp = { sign, exp[7:0]+127, man[6:0], 16'h0000 };
  assign fp = { sign, exp[7:0]+7'd127, man[6:0], 16'h0000 };

endmodule
