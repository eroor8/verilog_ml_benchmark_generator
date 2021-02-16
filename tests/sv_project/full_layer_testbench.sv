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
