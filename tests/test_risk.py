import numpy as np
import pandas as pd

from bank_volatility.risk import (
    christoffersen_cc_test,
    christoffersen_ind_test,
    kupiec_uc_test,
    standardized_student_t_quantile,
)


def test_standardized_student_t_quantile_is_negative_on_left_tail():
    assert standardized_student_t_quantile(0.01, 5.0) < 0


def test_coverage_tests_return_probabilities():
    hits = pd.Series([0] * 95 + [1] * 5)
    for fn, args in [
        (kupiec_uc_test, (hits, 0.05)),
        (christoffersen_ind_test, (hits,)),
        (christoffersen_cc_test, (hits, 0.05)),
    ]:
        _, pvalue = fn(*args)
        assert np.isnan(pvalue) or 0 <= pvalue <= 1
