import pandas as pd
import numpy as np
import pytest
from datetime import datetime
from SPyChart.SPyChart import SPC


def create_sample_data():
    dates = pd.Series(pd.date_range(start='2023-01-01', periods=30, freq='D'), name="date")
    values = np.random.rand(30) * 100
    return pd.DataFrame(values, index=dates, columns=['Value'])


def test_initialization():
    data = create_sample_data()
    spc = SPC(data_in=data, target_col='Value')

    assert spc.data_in.equals(data)
    assert spc.target_col == 'Value'
    assert spc.chart_type == 'Individual-chart'
    assert spc.change_dates is None
    assert spc.rules_table is not None
    assert isinstance(spc.rules_table, pd.DataFrame)
    assert 'Rule 1' in spc.rules_table.index

def test_clean_time_series_data():
    data = create_sample_data()
    data.iloc[5, 0] = np.nan  # Introduce a missing value
    spc = SPC(data_in=data, target_col='Value')

    spc._clean_time_series_data()

    missing_values = data.isnull().sum()
    assert missing_values.any()

def test_setup_single_run():
    data = create_sample_data()
    spc = SPC(data_in=data, target_col='Value')

    formatted_x_out, formatted_y_out = spc._setup_single_run(data)

    assert formatted_x_out is not None
    assert 'cl' in formatted_x_out.columns
    assert 'ucl' in formatted_x_out.columns
    assert 'lcl' in formatted_x_out.columns

def test_rules_func():
    data = create_sample_data()
    data['ucl'] = data['Value'].mean() + 3 * data['Value'].std()
    data['lcl'] = data['Value'].mean() - 3 * data['Value'].std()
    data['cl'] = data['Value'].mean()
    data['+2sd'] = data['Value'].mean() + 2 * data['Value'].std()
    data['-2sd'] = data['Value'].mean() - 2 * data['Value'].std()
    data['+1sd'] = data['Value'].mean() + data['Value'].std()
    data['-1sd'] = data['Value'].mean() - data['Value'].std()

    spc = SPC(data_in=data, target_col='Value')

    violations = spc._rules_func(data, 'Value')

    assert isinstance(violations, dict)
    assert 'Rule 1 violation' in violations
    assert 'Rule 2 violation' in violations
    assert 'Rule 3 violation' in violations


if __name__ == "__main__":
    pytest.main()
