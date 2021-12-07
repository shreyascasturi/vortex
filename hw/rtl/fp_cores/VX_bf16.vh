
`define vx_bf16_is_inf(raw)  (raw[30:23] == 8'hFF && raw[22:16] == 7'b0) 
`define vx_bf16_is_nan(raw)  (raw[30:23] == 8'hFF && raw[22:16] != 7'b0) 
`define vx_bf16_is_zero(raw) (raw[30:23] == 8'h00 && raw[22:16] == 7'b0) 

