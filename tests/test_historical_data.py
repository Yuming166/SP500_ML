from pathlib import Path

import pandas as pd
from openpyxl import Workbook

from sp500_forecastability.historical_data import _read_wind_excel
from sp500_forecastability.historical_replay_v2 import _inverse_loss_weights


def test_read_wind_excel_keeps_only_dated_data_rows(tmp_path: Path) -> None:
    path = tmp_path / "wind.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    for _ in range(31):
        sheet.append([None])
    sheet.append(["指标名称", "total", "index"])
    sheet.append(["时间区间", "2020-01-01:2020-01-02", "2020-01-01:2020-01-02"])
    sheet.append(["2020-01-01", 1.0, 2.0])
    sheet.append(["2020-01-02", 3.0, 4.0])
    sheet.append(["数据来源：Wind", None, None])
    workbook.save(path)

    actual = _read_wind_excel(path, ("total", "index"))

    assert actual.index.strftime("%Y-%m-%d").tolist() == ["2020-01-01", "2020-01-02"]
    assert actual.columns.tolist() == ["total", "index"]
    assert actual.loc["2020-01-02", "index"] == 4.0


def test_inverse_loss_weights_favour_lower_loss() -> None:
    weights = _inverse_loss_weights(pd.Series({"good": 0.1, "bad": 0.4}))

    assert weights.notna().all()
    assert weights["good"] > weights["bad"]
    assert weights.sum() == 1.0
