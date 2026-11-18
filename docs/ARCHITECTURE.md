# Architecture
Provider/import adapters -> schema validation -> append-only DuckDB/Parquet -> canonical contract mapping -> point-in-time features -> CLV models and separate game model -> execution-cost analysis -> shared service layer -> Streamlit and MCP.

Line CLV measures number improvement. Price CLV measures price/probability improvement on a matched contract. Fair probability is an outcome estimate. Net EV requires a fair probability and executable odds/costs. Execution risk captures stale quotes, limits, slippage and fill uncertainty. These are separate quantities.

The existing ATS model is integrated through versioned exports, not hardwired imports. Dashboard and MCP must call the same service layer.
