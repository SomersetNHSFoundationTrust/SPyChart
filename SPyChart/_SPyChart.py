'''
  █████████  ███████████               █████████  █████                           █████
 ███░░░░░███░░███░░░░░███             ███░░░░░███░░███                           ░░███
░███    ░░░  ░███    ░███ █████ ████ ███     ░░░  ░███████    ██████   ████████  ███████
░░█████████  ░██████████ ░░███ ░███ ░███          ░███░░███  ░░░░░███ ░░███░░███░░░███░
 ░░░░░░░░███ ░███░░░░░░   ░███ ░███ ░███          ░███ ░███   ███████  ░███ ░░░   ░███
 ███    ░███ ░███         ░███ ░███ ░░███     ███ ░███ ░███  ███░░███  ░███       ░███ ███
░░█████████  █████        ░░███████  ░░█████████  ████ █████░░████████ █████      ░░█████
 ░░░░░░░░░  ░░░░░          ░░░░░███   ░░░░░░░░░  ░░░░ ░░░░░  ░░░░░░░░ ░░░░░        ░░░░░
                           ███ ░███
                          ░░██████
                           ░░░░░░
'''

# --------------------------
# -** REQUIRED LIBRARIES **-
# --------------------------

import pandas as pd
import numpy as np
import json
import os
from numpy.lib.stride_tricks import sliding_window_view
'''

------------------
-** REFERENCES **-
------------------

- https://www.spcforexcel.com/knowledge/control-chart-basics/control-chart-rules-interpretation 
- https://www.spcforexcel.com/knowledge/control-chart-basics/applying-out-of-control-tests
- https://qi.elft.nhs.uk/wp-content/uploads/2018/10/Mohammed-et-al-2008-Plotting-basic-control-charts.pdf
- https://www.england.nhs.uk/improvement-hub/wp-content/uploads/sites/44/2017/11/A-guide-to-creating-and-interpreting-run-and-control-charts.pdf
- https://www.isdscotland.org/health-topics/quality-indicators/statistical-process-control/_docs/Statistical-Process-Control-Tutorial-Guide-180713.pdf
    
'''


class SPC:

    def __init__(self, df, target_col, chart_type='Individual-chart', change_dates=None, baseline_period=None):

        """

        This class does the following SPC analysis steps:

            - Takes in pandas dataframe, with a datetime index and the target column.
            (note, some charts require a third column ("p-chart" & "u-chart"), giving the sample size. 
            This column must be named 'n').

            - Calculates control lines (for the specified SPC chart).

            - Evaluates the data alongside 5 rules, which detects potential special cause variation.

            - Returns an interactive SPC chart (in Plotly) and the data needed to build your own chart.

        NOTE: You should have >= 20 data points in your dataset to use this tool.

        Args:

            df (pandas.DataFrame): Data to analyse.

            target_col (str): Name of target column.

            chart_type (str): Choose from:

                 - "XmR-chart"
                 - "Individual-chart"
                 - "p-chart"
                 - "np-chart"
                 - "c-chart"
                 - "u-chart"
                 - "XbarR-chart"
                 - "XbarS-chart"

             change_dates (list of str) (OPTIONAL): List of dates, which each represent a change in the underlying
             process being analysed. Control lines will be re-calculated after each date in the list.

             baseline_period (str) (OPTIONAL): Data before this date will be used to calculate the control lines.

         Returns:
            spc_data (

         SPC Rules:
         Rule 1: 1 point outside the +/- 3 sigma limits.
         Rule 2: 8 successive consecutive points above (or below) the centre line.
         Rule 3: 6 or more consecutive points steadily increasing or decreasing.
         Rule 4: 2 out of 3 successive points beyond +/- 2 sigma limits.
         Rule 5: 15 consecutive points within +/- 1 sigma on either side of the centre line.


        """

        self.df = df.copy()
        self.target_col = target_col
        self.chart_type = chart_type
        self.change_dates = change_dates
        self.baseline_period = pd.to_datetime(baseline_period)
        self.rules_table = None
        self.spc_data = None

        self._data_y = None
        self._data_x = None
        self._rules_list_x = None
        self._rules_list_y = None
        self._dict_rules_x = None
        self._dict_rules_y = None
        self._formatted_data_x = None
        self._formatted_data_y = None
        self._target_col_x = None
        self._target_col_y = None
        self._chart_name_x = None
        self._chart_name_y = None
        # Assuming index is date

        self._date_col = df.index.name

        base_dir = os.path.dirname(__file__)
        self.rules = json.load(open(os.path.join(base_dir, "rules.json")))


        # https://www.england.nhs.uk/improvement-hub/wp-content/uploads/sites/44/2017/11/A-gu
        # ide-to-creating-and-interpreting-run-and-control-charts.pdf
        rules_df = pd.DataFrame()
        rules_df['Rules'] = ['Rule 1', 'Rule 2', 'Rule 3', 'Rule 4', 'Rule 5']
        rules_df['Rule definition'] = ['1 point outside the +/- 3 sigma limits',
                                       '8 successive consecutive points above (or below) the centre line',
                                       '6 or more consecutive points steadily increasing or decreasing',
                                       '2 out of 3 successive points beyond +/- 2 sigma limits',
                                       '15 consecutive points within +/- 1 sigma on either side of the centre line']
        rules_df = rules_df.set_index('Rules')

        # Save rules attribute as dataframe.
        self.rules_table = rules_df

        # ------------------------------------------------------
        # -** Checking data/alerting user of potential issues.**-
        # ------------------------------------------------------

        if (df.index.value_counts() > 1).any():
            print('Duplicate dates detected.')

            if df.index.value_counts().min() == df.index.value_counts().max():
                print(f'Constant sample size = {df.index.value_counts()[0]}')
                self.sample_size = df.index.value_counts()[0]

        if len(df) <= 20:
            print('Less than 20 data points detected. Consider collecting more data before using this tool.')

        if self.baseline_period is not None:
            if len(df.copy().loc[:self.baseline_period]) < 20:
                print('Less than 20 data points detected in baseline period. Consider adding more \
                data pre-baseline.')

        self.setup()
        self.check_rules()


    # -----------------------
    # -** MAIN CLASS CODE **-
    # -----------------------

    def run_spc(self):

        """
	    Run all SPC methods in a convenient function.

	    """

        self.setup()
        self.check_rules()

        return self.spc_data

    def setup(self):
        """

        Function enabling multiple runs of the _setup_single_run() method. This is to allow
        for control lines to be calculated multiple times (if specified by user using the change_dates parameter).

        If change_dates is not specified, it runs only once.


        """
        # Firstly, we check for any DQ issues using _clean_time_series_data().
        self._clean_time_series_data()
        self.__format_data()

    def _clean_time_series_data(self):

        """
        Checks data for any data quality issues.

        Inputs:
            data (pandas.DataFrame): Data to analyse (with datetime index).
        """

        # Check if index is in pandas datetime format
        if pd.api.types.is_datetime64_any_dtype(self.df.index):
            try:
                self.df.index = pd.to_datetime(self.df.index)
            except ValueError:
                raise ValueError(f"Index not in required format. Please use datetime index.")

        # Check for missing data
        missing_values = self.df.isnull().sum()

        if missing_values.any():
            print("Missing values detected:")
            print(missing_values)

    def __format_data(self):
        # TODO: Docstring
        # Check input arguments to determine number of runs of the _setup_single_run() method.
        if self.change_dates is None:
            self._formatted_data_x, self._formatted_data_y = self._setup_single_run(data=self.df)

        else:
            list_dates = [self.df.index[0]] + self.change_dates + [self.df.index[-1]]
            list_dataframes = [
                                  self.df.loc[list_dates[idx]: (pd.to_datetime(list_dates[idx + 1]) - pd.DateOffset(days=1))]
                                  for idx in range(0, len(list_dates) -2)
                              ] + [self.df.loc[list_dates[-2]:(pd.to_datetime(list_dates[-1]))]]

            formatted_x = []
            formatted_y = []
            for data in list_dataframes:
                formatted_x_out, formatted_y_out = self._setup_single_run(data=data)
                formatted_x.append(formatted_x_out)
                formatted_y.append(formatted_y_out)

            self._formatted_data_x = pd.concat(formatted_x)
            self._formatted_data_y = None if all(x is None for x in formatted_y) else pd.concat(formatted_y)



    def _setup_single_run(self, data):

        """

       Type of SPC chart currently available:
       "XmR-chart", "Individual-chart", "p-chart", "np-chart",  "c-chart", "u-chart", "XbarR-chart", XbarS-chart"

       !! Can add chart types as requested.

       Returns dataframe with control limits, upper/lower as well zones A, B, C which are the 3 zones between the
       upper/lower limits and the center line.

        Args:
            df (pandas.DataFrame): Data input (with datetime index).
            chart_type (str): SPC chart type (selected from one of the above).

        Returns:
            self.pandas.DataFrame: Pandas dataframe with control limits for each chart.

       """

        # Reference dictionary for X_bar chart control limits calculations. Different sample size will result in
        # different control limits for the xbar charts.

        x_bar_vals = {'A2': {2: '1.88', 3: '1.023', 4: '0.729', 5: '0.577', 6: '0.483', 7: '0.419',
                             8: '0.373', 9: '0.337', 10: '0.308', 11: '0.285', 12: '0.266',
                             13: '0.249', 14: '0.235', 15: '0.223', 16: '0.212', 17: '0.203',
                             18: '0.194', 19: '0.187', 20: '0.18', 21: '0.173', 22: '0.167',
                             23: '0.162', 24: '0.157', 25: '0.153'},
                      'A3': {2: '2.659', 3: '1.954', 4: '1.628', 5: '1.427', 6: '1.287', 7: '1.182',
                             8: '1.099', 9: '1.032', 10: '0.975', 11: '0.927', 12: '0.886', 13: '0.85',
                             14: '0.817', 15: '0.789', 16: '0.763', 17: '0.739', 18: '0.718',
                             19: '0.698', 20: '0.68', 21: '0.663', 22: '0.647', 23: '0.633',
                             24: '0.619', 25: '0.606'},
                      'd2': {2: '1.128', 3: '1.693', 4: '2.059', 5: '2.326', 6: '2.534', 7: '2.704',
                             8: '2.847', 9: '2.97', 10: '3.078', 11: '3.173', 12: '3.258', 13: '3.336',
                             14: '3.407', 15: '3.472', 16: '3.532', 17: '3.588', 18: '3.64', 19: '3.689',
                             20: '3.735', 21: '3.778', 22: '3.819', 23: '3.858', 24: '3.895',
                             25: '3.931'},
                      'D3': {2: '0', 3: '0', 4: '0', 5: '0', 6: '0', 7: '0.076', 8: '0.136',
                             9: '0.184', 10: '0.223', 11: '0.256', 12: '0.283', 13: '0.307',
                             14: '0.328', 15: '0.347', 16: '0.363', 17: '0.378', 18: '0.391',
                             19: '0.403', 20: '0.415', 21: '0.425', 22: '0.434', 23: '0.443',
                             24: '0.451', 25: '0.459'},
                      'D4': {2: '3.267', 3: '2.574', 4: '2.282', 5: '2.114', 6: '2.004', 7: '1.924',
                             8: '1.864', 9: '1.816', 10: '1.777', 11: '1.744', 12: '1.717',
                             13: '1.693', 14: '1.672', 15: '1.653', 16: '1.637', 17: '1.622',
                             18: '1.608', 19: '1.597', 20: '1.585', 21: '1.575', 22: '1.566',
                             23: '1.557', 24: '1.548', 25: '1.541'},
                      'B3': {2: '0', 3: '0', 4: '0', 5: '0', 6: '0.03', 7: '0.118', 8: '0.185',
                             9: '0.239', 10: '0.284', 11: '0.321', 12: '0.354', 13: '0.382',
                             14: '0.406', 15: '0.428', 16: '0.448', 17: '0.466', 18: '0.482',
                             19: '0.497', 20: '0.51', 21: '0.523', 22: '0.534', 23: '0.545',
                             24: '0.555', 25: '0.565'},
                      'B4': {2: '3.267', 3: '2.568', 4: '2.266', 5: '2.089', 6: '1.97',
                             7: '1.882', 8: '1.815', 9: '1.761', 10: '1.716', 11: '1.679',
                             12: '1.646', 13: '1.618', 14: '1.594', 15: '1.572', 16: '1.552',
                             17: '1.534', 18: '1.518', 19: '1.503', 20: '1.49', 21: '1.477',
                             22: '1.466', 23: '1.455', 24: '1.445', 25: '1.435'}}

        # If we're not using baseline data, we set baseline data to last value of input data.
        if self.change_dates is not None:
            self.baseline_period = data.index[-1]

        if (self.baseline_period is None) & (self.change_dates is None):
            self.baseline_period = data.index[-1]

        if (self.chart_type == 'Individual-chart') or (self.chart_type == 'XmR-chart'):

            baseline_data = data.copy().loc[:self.baseline_period, :]
            baseline_data['mR'] = data[self.target_col].diff().abs().loc[:self.baseline_period]

            # Individual chart
            data_I = data.copy()
            data_I['cl'] = baseline_data[self.target_col].mean()
            data_I['lcl'] = baseline_data[self.target_col].mean() - 3 * (
                    (baseline_data['mR'].iloc[1:len(baseline_data['mR'])]) /
                    1.128).mean()
            data_I['ucl'] = baseline_data[self.target_col].mean() + 3 * (
                    (baseline_data['mR'].iloc[1:len(baseline_data['mR'])]) /
                    1.128).mean()
            data_I['+1sd'] = baseline_data[self.target_col].mean() + 1 * (
                    (baseline_data['mR'].iloc[1:len(baseline_data['mR'])]) /
                    1.128).mean()
            data_I['-1sd'] = baseline_data[self.target_col].mean() - 1 * (
                    (baseline_data['mR'].iloc[1:len(baseline_data['mR'])]) /
                    1.128).mean()
            data_I['+2sd'] = baseline_data[self.target_col].mean() + 2 * (
                    (baseline_data['mR'].iloc[1:len(baseline_data['mR'])]) /
                    1.128).mean()
            data_I['-2sd'] = baseline_data[self.target_col].mean() - 2 * (
                    (baseline_data['mR'].iloc[1:len(baseline_data['mR'])]) /
                    1.128).mean()

            # mR chart
            data_mR = data.copy()
            data_mR['r'] = data[self.target_col].diff().abs().values
            data_mR['cl'] = baseline_data['mR'].mean()
            data_mR['lcl'] = 0
            data_mR['ucl'] = baseline_data['mR'].mean() + 3.27 * (
                baseline_data['mR'].iloc[1:len(baseline_data['mR'])].mean())

            zone = (data_mR['ucl'] - data_mR['cl']).mean()

            data_mR['+1sd'] = baseline_data['mR'].mean() - 1 * zone
            data_mR['-1sd'] = baseline_data['mR'].mean() + 1 * zone
            data_mR['+2sd'] = baseline_data['mR'].mean() - 2 * zone
            data_mR['-2sd'] = baseline_data['mR'].mean() + 2 * zone
            # These are probably not correctly calculated, but for consistency have kept in for now.

            # Check lcl doesn't fall below 0.
            if data_mR['lcl'][0] < 0:
                data_mR['lcl'] = 0

            if self.chart_type == 'Individual-chart':
                return data_I, None

            elif self.chart_type == 'XmR-chart':
                return data_I, data_mR


        elif self.chart_type == 'XbarR-chart':

            data_x_bar = data.copy().reset_index(drop=False)
            x_bar_df = data_x_bar.groupby(by=self._date_col).mean().reset_index(drop=False).rename(
                columns={self.target_col: 'x_bar'})
            x_bar_df['r'] = data_x_bar.groupby(by=self._date_col).max()[self.target_col].values - \
                            data_x_bar.groupby(by=self._date_col).min()[
                                self.target_col].values

            df_r = x_bar_df[['r', self._date_col]].set_index(self._date_col,
                                                             drop=True).rename(columns={'x_bar': self.target_col})
            df_out = x_bar_df[['x_bar', self._date_col]].set_index(self._date_col,
                                                                   drop=True).rename(columns={'x_bar': self.target_col})
            df_out['cl'] = df_out.loc[:self.baseline_period][self.target_col].mean()
            df_out['lcl'] = df_out.loc[:self.baseline_period][self.target_col].mean() - float(
                x_bar_vals['A2'][self.sample_size]) * \
                            df_r.loc[:self.baseline_period]['r'].mean()
            df_out['ucl'] = df_out.loc[:self.baseline_period][self.target_col].mean() + float(
                x_bar_vals['A2'][self.sample_size]) * \
                            df_r.loc[:self.baseline_period]['r'].mean()

            # Value to get each zone (A, B, C)
            zone = ((df_out['ucl'] - df_out['cl']) / 3)[0]

            df_out['+1sd'] = df_out['cl'] + zone
            df_out['-1sd'] = df_out['cl'] - zone
            df_out['+2sd'] = df_out['cl'] + 2 * zone
            df_out['-2sd'] = df_out['cl'] - 2 * zone

            # Check lcl doesn't fall below 0.
            df_out['lcl'] = [x if x > 0 else 0 for x in df_out['lcl']]

            df_out_R = pd.DataFrame()
            df_out_R[self._date_col] = x_bar_df[self._date_col].values
            df_out_R['r'] = x_bar_df['r'].values
            df_out_R['cl'] = df_r.loc[:self.baseline_period]['r'].mean()
            df_out_R['lcl'] = df_r.loc[:self.baseline_period]['r'].mean() * float(
                x_bar_vals['D3'][self.sample_size])
            df_out_R['ucl'] = df_r.loc[:self.baseline_period]['r'].mean() * float(
                x_bar_vals['D4'][self.sample_size])

            zone_R = ((df_out_R['ucl'] - df_out_R['cl']) / 3)[0]

            df_out_R['+1sd'] = df_out_R['cl'] + zone_R
            df_out_R['-1sd'] = df_out_R['cl'] - zone_R
            df_out_R['+2sd'] = df_out_R['cl'] + 2 * zone_R
            df_out_R['-2sd'] = df_out_R['cl'] - 2 * zone_R

            df_out_R = df_out_R.set_index(self._date_col, drop=True)

            if df_out_R['lcl'][0] < 0:
                df_out_R['lcl'] = 0

            return df_out, df_out_R

        elif self.chart_type == 'XbarS-chart':

            data_x_bar = data.copy().reset_index(drop=False)

            x_bar_df = data_x_bar.groupby(by=self._date_col).mean().reset_index(drop=False).rename(
                columns={self.target_col: 'x_bar'})
            x_bar_df['r'] = data_x_bar.groupby(by=self._date_col).max()[self.target_col].values - \
                            data_x_bar.groupby(by=self._date_col).min()[
                                self.target_col].values

            df_r = x_bar_df[['r', self._date_col]].set_index(self._date_col, drop=True).rename(
                columns={'x_bar': self.target_col})
            df_out = x_bar_df[['x_bar', self._date_col]].set_index(self._date_col,
                                                                   drop=True).rename(columns={'x_bar': self.target_col})
            df_out['cl'] = df_out.loc[:self.baseline_period][self.target_col].mean()
            df_out['lcl'] = df_out.loc[:self.baseline_period][self.target_col].mean() - float(
                x_bar_vals['A3'][self.sample_size]) * \
                            df_r.loc[:self.baseline_period]['r'].mean()
            df_out['ucl'] = df_out.loc[:self.baseline_period][self.target_col].mean() + float(
                x_bar_vals['A3'][self.sample_size]) * \
                            df_r.loc[:self.baseline_period]['r'].mean()

            zone = ((df_out['ucl'] - df_out['cl']) / 3)

            df_out['+1sd'] = df_out['cl'] + zone
            df_out['-1sd'] = df_out['cl'] - zone
            df_out['+2sd'] = df_out['cl'] + 2 * zone
            df_out['-2sd'] = df_out['cl'] - 2 * zone

            # Check lcl doesn't fall below 0.
            df_out['lcl'] = [x if x > 0 else 0 for x in df_out['lcl']]

            df_out_S = pd.DataFrame()
            df_out_S[self._date_col] = x_bar_df[self._date_col].values
            df_out_S['r'] = x_bar_df['r'].values
            df_out_S['cl'] = df_r.loc[:self.baseline_period]['r'].mean()
            df_out_S['lcl'] = df_r.loc[:self.baseline_period]['r'].mean() * float(
                x_bar_vals['B3'][self.sample_size])
            df_out_S['ucl'] = df_r.loc[:self.baseline_period]['r'].mean() * float(
                x_bar_vals['B4'][self.sample_size])

            zone_R = ((df_out_S['ucl'] - df_out_S['cl']) / 3)

            df_out_S['+1sd'] = df_out_S['cl'] + zone_R
            df_out_S['-1sd'] = df_out_S['cl'] - zone_R
            df_out_S['+2sd'] = df_out_S['cl'] + 2 * zone_R
            df_out_S['-2sd'] = df_out_S['cl'] - 2 * zone_R

            # Check lcl doesn't fall below 0.
            if df_out_S['lcl'][0] < 0:
                df_out_S['lcl'] = 0

            df_out_S = df_out_S.set_index(self._date_col, drop=True)

            return df_out, df_out_S

        elif self.chart_type == 'c-chart':

            df = data.copy()

            df['cl'] = df.loc[:self.baseline_period][self.target_col].mean()
            df['lcl'] = df.loc[:self.baseline_period][self.target_col].mean() - \
                        (3 * ((df[self.target_col].mean()) ** 0.5))
            df['ucl'] = df.loc[:self.baseline_period][self.target_col].mean() + \
                        (3 * ((df[self.target_col].mean()) ** 0.5))

            df['+1sd'] = df.loc[:self.baseline_period][self.target_col].mean() + \
                         (1 * ((df[self.target_col].mean()) ** 0.5))
            df['-1sd'] = df.loc[:self.baseline_period][self.target_col].mean() - \
                         (1 * ((df[self.target_col].mean()) ** 0.5))
            df['+2sd'] = df.loc[:self.baseline_period][self.target_col].mean() + \
                         (2 * ((df[self.target_col].mean()) ** 0.5))
            df['-2sd'] = df.loc[:self.baseline_period][self.target_col].mean() - \
                         (2 * ((df[self.target_col].mean()) ** 0.5))

            # Check lcl doesn't fall below 0.
            df['lcl'] = [x if x > 0 else 0 for x in df['lcl']]

            return df, None

        elif self.chart_type == 'p-chart':

            df = data.copy()

            df[self.target_col] = df[self.target_col] / df['n']

            df['cl'] = df.loc[:self.baseline_period][self.target_col].mean()
            df['lcl'] = df.loc[:self.baseline_period][self.target_col].mean() - 3 * (
                    (df[self.target_col].mean() *
                     (1 - df.loc[:self.baseline_period][self.target_col].mean())) /
                    df.loc[:self.baseline_period]['n']) ** 0.5

            df['ucl'] = df.loc[:self.baseline_period][self.target_col].mean() + 3 * (
                    (df[self.target_col].mean() * (
                            1 - df.loc[:self.baseline_period][self.target_col].mean())) /
                    df.loc[:self.baseline_period]['n']) ** 0.5

            df['+1sd'] = df.loc[:self.baseline_period][self.target_col].mean() + 1 * (
                    (df[self.target_col].mean() *
                     (1 - df.loc[:self.baseline_period][self.target_col].mean())) /
                    df.loc[:self.baseline_period]['n']) ** 0.5

            df['-1sd'] = df.loc[:self.baseline_period][self.target_col].mean() - 1 * (
                    (df[self.target_col].mean() * (
                            1 - df.loc[:self.baseline_period][self.target_col].mean())) /
                    df.loc[:self.baseline_period]['n']) ** 0.5

            df['+2sd'] = df.loc[:self.baseline_period][self.target_col].mean() + 2 * (
                    (df[self.target_col].mean() *
                     (1 - df.loc[:self.baseline_period][self.target_col].mean())) /
                    df.loc[:self.baseline_period]['n']) ** 0.5

            df['-2sd'] = df.loc[:self.baseline_period][self.target_col].mean() - 2 * (
                    (df[self.target_col].mean() * (
                            1 - df.loc[:self.baseline_period][self.target_col].mean())) /
                    df.loc[:self.baseline_period]['n']) ** 0.5

            # Check lcl doesn't fall below 0.
            df['lcl'] = [x if x > 0 else 0 for x in df['lcl']]

            return df, None

        elif self.chart_type == 'np-chart':

            df = data.copy()

            p = df[self.target_col].loc[:self.baseline_period].sum() \
                / df.loc[:self.baseline_period]['n'].sum()

            df['cl'] = df.loc[:self.baseline_period][self.target_col].mean()

            df['ucl'] = df.loc[:self.baseline_period][self.target_col].mean() \
                        + 3 * (df.loc[:self.baseline_period][self.target_col].mean() * (
                    1 - p)) ** 0.5

            df['lcl'] = df.loc[:self.baseline_period][self.target_col].mean() \
                        - 3 * (df.loc[:self.baseline_period][self.target_col].mean() * (
                    1 - p)) ** 0.5

            df['+1sd'] = df.loc[:self.baseline_period][self.target_col].mean() \
                         + 1 * (df.loc[:self.baseline_period][self.target_col].mean() * (
                    1 - p)) ** 0.5
            df['-1sd'] = df.loc[:self.baseline_period][self.target_col].mean() \
                         - 1 * (df.loc[:self.baseline_period][self.target_col].mean() * (
                    1 - p)) ** 0.5

            df['+2sd'] = df.loc[:self.baseline_period][self.target_col].mean() \
                         + 2 * (df.loc[:self.baseline_period][self.target_col].mean() * (
                    1 - p)) ** 0.5
            df['-2sd'] = df.loc[:self.baseline_period][self.target_col].mean() \
                         - 2 * (df.loc[:self.baseline_period][self.target_col].mean() * (
                    1 - p)) ** 0.5

            # Check lcl doesn't fall below 0.
            df['lcl'] = [x if x > 0 else 0 for x in df['lcl']]

            return df, None

        elif self.chart_type == 'u-chart':

            df = data.copy()

            df[self.target_col] = df[self.target_col] / df['n']

            df['cl'] = df.loc[:self.baseline_period][self.target_col].mean()
            df['lcl'] = df.loc[:self.baseline_period][self.target_col].mean() - 3 * \
                        (((df.loc[:self.baseline_period][self.target_col].mean())) /
                         (df.loc[:self.baseline_period]['n'])) ** 0.5
            df['ucl'] = df.loc[:self.baseline_period][self.target_col].mean() + 3 * \
                        (((df.loc[:self.baseline_period][self.target_col].mean())) /
                         (df.loc[:self.baseline_period]['n'])) ** 0.5

            df['+1sd'] = df.loc[:self.baseline_period][self.target_col].mean() + 1 * \
                         (((df.loc[:self.baseline_period][self.target_col].mean())) /
                          (df.loc[:self.baseline_period]['n'])) ** 0.5

            df['-1sd'] = df.loc[:self.baseline_period][self.target_col].mean() - 1 * \
                         (((df.loc[:self.baseline_period][self.target_col].mean())) /
                          (df.loc[:self.baseline_period]['n'])) ** 0.5

            df['+2sd'] = df.loc[:self.baseline_period][self.target_col].mean() + 2 * \
                         (((df.loc[:self.baseline_period][self.target_col].mean())) /
                          (df.loc[:self.baseline_period]['n'])) ** 0.5

            df['-2sd'] = df.loc[:self.baseline_period][self.target_col].mean() - 2 * \
                         (((df.loc[:self.baseline_period][self.target_col].mean())) /
                          (df.loc[:self.baseline_period]['n'])) ** 0.5

            # Check lcl doesn't fall below 0.
            df['lcl'] = [x if x > 0 else 0 for x in df['lcl']]

            return df, None

        else:
            print('Chart type must be one of "XmR-chart", "Individual-chart", "p-chart",  '
                  '"np-chart", "c-chart", "u-chart", "XbarR-chart", XbarS-chart"')

    def check_rules(self):
        """

        Requires setup() method to be called first.

        Checks the calculated control lines, and tests up to 8 rules.

        """
        self._target_col_x = self.target_col
        self._target_col_y = 'r' if self.chart_type in ("XmR-chart", "XbarR-chart", "XbarS-chart") else None

        # Check rules for both graphs (checking second graph data is not None)
        self._dict_rules_x = self._rules_func(self._formatted_data_x, self._target_col_x)
        self._dict_rules_y = None if self._formatted_data_y is None else self._rules_func(self._formatted_data_y, self._target_col_y)

        rules = self.rules[self.chart_type]
        self._rules_list_x = rules["rules"]
        self._chart_name_x = rules.get("chart_name_x")
        self._chart_name_y = rules.get("chart_name_y")

        df_x = self.__identify_rule_violations_x()
        df_y = self.__identify_rule_violations_y() if self._dict_rules_y is not None else None

        output_data = pd.concat([df_x, df_y])
        output_data.rename(columns={self.target_col: 'target'}, inplace=True)

        self.spc_data = output_data

    def _rules_func(self, input_df, target_col):
        """
        Checks up to 5 SPC rules. Not all rules are suitable for all charts, therefore, fewer rules will
        be tested in these instances.

        !! Can add rules as requested.

        Args:
            input_df (pandas.DataFrame): Data to analyse.
            target_col (str): Name of target column.

        Returns:
            dict: Dictionary of rule violations with Rule 1-5 as keys, and dates as values.

        """
        violations = {}

        # Rule 1: Point outside the +/- 3 sigma limits
        rule1 = (input_df[target_col] > input_df['ucl']) | (input_df[target_col] < input_df['lcl'])
        violations['Rule 1 violation'] = input_df.index[rule1].tolist()

        # Rule 2: 8 successive consecutive points above (or below) the centre line
        rule2 = self.__calculate_rule_violations(
            input_df[target_col] > input_df['cl'], input_df[target_col] < input_df['cl'],
            input_df=input_df, window_size=8
        )
        violations['Rule 2 violation'] = rule2

        # Rule 3: 6 or more consecutive points steadily increasing or decreasing
        rule3 = self.__calculate_rule_violations(
            np.diff(input_df[target_col]) > 0, np.diff(input_df[target_col]) < 0,
            input_df=input_df, window_size=6, buffer=1,
        )
        violations['Rule 3 violation'] = rule3

        # Rule 4: 2 out of 3 successive points beyond +/- 2 sigma limits
        # TODO: Can this be manipulated to go through the function?
        rule4 = []
        for i in range(2, len(input_df)):
            subset = input_df.iloc[i - 2:i + 1]
            if ((subset[target_col] > subset['+2sd']).sum() >= 2) or ((subset[target_col] < subset['-2sd']).sum() >= 2):
                rule4.append(input_df.index[i])
        violations['Rule 4 violation'] = rule4

        # Rule 5: 15 consecutive points within +/- 1 sigma on either side of the centre line
        rule5 = self.__calculate_rule_violations(
            np.abs(input_df[target_col] - input_df['cl']) <= (input_df['+1sd'] - input_df['cl']),
            input_df=input_df, window_size=15
        )
        violations['Rule 5 violation'] = rule5

        return violations

    def __calculate_rule_violations(self, *args, input_df, window_size, buffer=0):
        # TODO: docstring

        mask = [False] * (window_size-1+buffer) + \
        np.array([sliding_window_view(arg, window_shape=window_size).all(axis=1) for arg in args]).any(axis=0).tolist()

        return input_df.index[mask]


    def __identify_rule_violations_x(self):
        dictionary_x = {key: value for key, value in self._dict_rules_x.items() if key in self._rules_list_x}
        df_x = self._formatted_data_x
        # Add columns with string headers and binary representation
        for header, date_list in dictionary_x.items():
            df_x[header] = df_x.reset_index()[self._date_col].apply(lambda x: 1 if x in date_list else 0).values

        df_x['chart type'] = self._chart_name_x

        return df_x

    def __identify_rule_violations_y(self):
        rules = ['Rule 1 violation', 'Rule 2 violation', 'Rule 3 violation']
        dictionary_y = {key: value for key, value in self._dict_rules_y.items() if key in rules}

        df_y = self._formatted_data_y

        for header, date_list in dictionary_y.items():
            df_y[header] = df_y.reset_index()[self._date_col].apply(lambda x: 1 if x in date_list else 0).values

        df_y['chart type'] = self._chart_name_y

        df_y.reset_index(inplace=True)

        return df_y


        output_data = pd.concat([self._data_x, self._data_y])
        output_data.rename(columns={self.target_col: 'target'}, inplace=True)
        self.spc_data = output_data
