`timescale 1ns/1ns

module full_layer_testbench();
    reg clk, reset, start;
    wire done;
   
    
    initial begin
        $display("Starting system verilog simulation");
        clk = 0;
        reset = 0;  
        start = 0;   
        #10
	reset = 1;
        #10
	reset = 0;
        #20
        start = 1;
        #10
	start = 0;
       wait(done);
       $display("Done asserted!")  ;
       
        #200
	$stop;
  
    end
    
    always
        #5 clk = ~clk;
    
    test_sv dut0
    (
      .clk(clk),
      .reset(reset) ,
      .sel(1'b0) ,
      .sm_start(start),
      .done(done)
    );

endmodule

module ml_block_weights
(
  input  logic  clk ,
  input  logic  clk_portain ,
  input  logic  clk_portaout ,
  input  logic  clr0 ,
  input  logic  clr1 ,
  input  logic  ena0 ,
  input  logic  ena1 ,
  input  logic  ena2 ,
  input  logic  ena3 ,
  input  logic [1:0] portaaddr ,
  input  logic  portaaddrstall ,
  input  logic  portabyteenamasks ,
  input  logic [127:0] portadatain ,
  output logic [127:0] portadataout ,
  input  logic  portare ,
  input  logic  portawe ,
  input  logic  reset 
);
   reg [127:0] 	Mem [0:(1<<2)-1];
   always @ (*) begin
      portadataout = Mem[portaaddr];
   end
   always @ (posedge clk) begin
      if (portawe) begin
         Mem[portaaddr] <= portadatain;	 
      end
   end
endmodule

module ml_block_input #(
    parameter ADDR_WIDTH = 8,
    parameter DATA_WIDTH = 16
		    )
(
  input  logic  clk ,
  input  logic  clk_portain ,
  input  logic  clk_portaout ,
  input  logic  clr0 ,
  input  logic  clr1 ,
  input  logic  ena0 ,
  input  logic  ena1 ,
  input  logic  ena2 ,
  input  logic  ena3 ,
  input  logic [ADDR_WIDTH-1:0] portaaddr ,
  input  logic  portaaddrstall ,
  input  logic  portabyteenamasks ,
  input  logic [DATA_WIDTH-1:0] portadatain ,
  output logic [DATA_WIDTH-1:0] portadataout ,
  input  logic  portare ,
  input  logic  portawe ,
  input  logic  reset 
);
   reg [15:0] 	Mem [0:(1<<ADDR_WIDTH)-1];
    
   initial begin
      for (integer i=0; i < 1<<ADDR_WIDTH; i++)
	Mem[i] <= 0;
   end
   always @ (*) begin
      portadataout = Mem[portaaddr];
   end
   always @ (posedge clk) begin
      if (portawe) begin
         Mem[portaaddr] <= portadatain;	 
      end
   end
endmodule


module emif_inner #(
    parameter ADDR_WIDTH = 14,
    parameter DATA_WIDTH = 128
		    )
(
  input  logic [ADDR_WIDTH-1:0] address ,
  input  logic  clk ,
  input  logic [DATA_WIDTH-1:0] datain ,
  output logic [DATA_WIDTH-1:0] dataout ,
  input  logic  reset ,
  input  logic  wen 
);
   reg [DATA_WIDTH-1:0] 	data [0:(1<<ADDR_WIDTH)-1];
   initial begin
      $readmemh("orig_emif_contents.mem", data);
   end
   always @ (*) begin
      dataout = data[address];
   end
   always @ (posedge clk) begin
      if (wen) begin
         data[address] <= datain;	 
      end
   end
endmodule

//module emif_inner 
//(
//  input  logic [7:0] address ,
//  input  logic  clk ,
//  input  logic [127:0] datain ,
//  output logic [127:0] dataout ,
//  input  logic  reset ,
//  input  logic  wen 
//);
//   reg [127:0] 	data [0:(1<<8)-1];
//   initial begin
//      $readmemh("orig_emif_contents.mem", data);
//   end
//   always @ (*) begin
//      dataout = data[address];
//   end
//   always @ (posedge clk) begin
//      if (wen) begin
//         data[address] <= datain;	 
//      end
//   end
//endmodule
