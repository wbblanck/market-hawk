import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from nfl_market.mcp_server.server import TRANSPORTS, frame_records, run, to_jsonable


def test_timestamps_serialize_as_iso_utc():
    f = pd.DataFrame({"t": [pd.Timestamp("2026-09-01T08:00", tz="America/New_York"), pd.NaT],
                      "n": [np.int64(3), None], "x": [1.5, float("nan")]})
    recs = frame_records(f)
    assert recs[0]["t"] == "2026-09-01T12:00:00Z" and recs[1]["t"] is None
    assert recs[0]["n"] == 3 and recs[1]["x"] is None
    json.dumps(recs)  # must be JSON-serialisable
    assert to_jsonable(datetime(2026, 9, 1, 12, tzinfo=timezone.utc)) == "2026-09-01T12:00:00Z"


def test_transports_declared_and_validated():
    assert TRANSPORTS == ("stdio", "sse", "streamable-http")
    with pytest.raises(ValueError, match="Unsupported"):
        run(transport="websocket")


def test_build_server_registers_tools(tmp_path):
    pytest.importorskip("mcp")
    from nfl_market.mcp_server.server import build_server
    from nfl_market.services.market_service import MarketService
    m = build_server(MarketService(str(tmp_path / "db.duckdb")))
    assert m is not None
