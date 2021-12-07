

module VX_bf16_rshft (
  input wire [31:0] value,
  input wire [15:0] n,

  output wire [31:0] shifted
);

  wire residue;

  // bitwise reduce all bits lost in the rightshift
  assign residue = | ((n >= 32) ? value : value & ((1 << n)-1));

  // shift, or-ing last bit with residue
  assign shifted = ((n >= 32) ? 0 : (value >> n)) | {31'b0, residue};

endmodule
