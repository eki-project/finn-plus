module dfx_finn_decouple_wrapper #(
	parameter integer  DATA_WIDTH   = 8,
	parameter integer  DECOUPLE_CNT = 1
)(
	input  wire                    aclk,
	input  wire                    aresetn,

	input  wire [DATA_WIDTH-1:0]   s_axis_tdata,
	input  wire                    s_axis_tvalid,
	output wire                    s_axis_tready,

	output wire [DATA_WIDTH-1:0]   m_axis_tdata,
	output wire                    m_axis_tvalid,
	input  wire                    m_axis_tready,

	input  wire                    decouple,
	output wire                    decouple_status
);

	dfx_finn_decouple #(
		.DATA_WIDTH(DATA_WIDTH),
		.DECOUPLE_CNT(DECOUPLE_CNT)
	) inst_dfx_finn_decouple (
		.aclk(aclk),
		.aresetn(aresetn),
		.s_axis_tdata(s_axis_tdata),
		.s_axis_tvalid(s_axis_tvalid),
		.s_axis_tready(s_axis_tready),
		.m_axis_tdata(m_axis_tdata),
		.m_axis_tvalid(m_axis_tvalid),
		.m_axis_tready(m_axis_tready),
		.decouple(decouple),
		.decouple_status(decouple_status)
	);

endmodule : dfx_finn_decouple_wrapper
