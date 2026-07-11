// skid_buffer: valid/ready pipeline decoupling register.
//
// Registers both data and the ready path (s_ready depends only on internal
// state, never combinationally on m_ready), sustaining full throughput.
// Capacity: 2 entries (output register + skid register).
module skid_buffer #(
    parameter int unsigned WIDTH = 32
) (
    input  logic             clk,
    input  logic             rst_n,

    // upstream
    input  logic             s_valid,
    output logic             s_ready,
    input  logic [WIDTH-1:0] s_data,

    // downstream
    output logic             m_valid,
    input  logic             m_ready,
    output logic [WIDTH-1:0] m_data
);

  logic             skid_valid;
  logic [WIDTH-1:0] skid_data;

  // Accept as long as the skid register is free.
  assign s_ready = !skid_valid;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      m_valid    <= 1'b0;
      skid_valid <= 1'b0;
    end else begin
      // Output register: refill whenever empty or draining this cycle.
      if (!m_valid || m_ready) begin
        if (skid_valid) begin
          m_valid <= 1'b1;
          m_data  <= skid_data;
        end else begin
          m_valid <= s_valid && s_ready;
          if (s_valid && s_ready) begin
            m_data <= s_data;
          end
        end
      end
      // Skid register: park the input beat when the output register is stalled.
      if (skid_valid) begin
        if (!m_valid || m_ready) begin
          skid_valid <= 1'b0;
        end
      end else if (s_valid && s_ready && m_valid && !m_ready) begin
        skid_valid <= 1'b1;
        skid_data  <= s_data;
      end
    end
  end

endmodule
