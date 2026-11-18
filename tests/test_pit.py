import pandas as pd
import pytest
from nfl_market.features.pit import assert_point_in_time
def test_future_feature_rejected():
    x=pd.DataFrame({"p":pd.to_datetime(["2026-09-01"],utc=True),"a":pd.to_datetime(["2026-09-02"],utc=True)})
    with pytest.raises(ValueError,match="leakage"): assert_point_in_time(x,"p","a")
