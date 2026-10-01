from pathlib import Path

import numpy as np

from bank_volatility.data import close_prices, compute_log_returns, load_finam_export


def test_load_finam_export_and_returns(tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_text(
        "<TICKER>;<PER>;<DATE>;<TIME>;<OPEN>;<HIGH>;<LOW>;<CLOSE>;<VOL>\n"
        "SBER;D;240101;000000;100;102;99;101;1000\n"
        "SBER;D;240102;000000;101;103;100;102;1100\n"
        "SBER;D;240103;000000;102;104;101;104;1200\n",
        encoding="utf-8",
    )

    frame = load_finam_export(path)
    prices = close_prices(frame)
    returns = compute_log_returns(prices)

    assert len(frame) == 3
    assert len(returns) == 2
    assert np.isclose(returns.iloc[0], np.log(102 / 101))
