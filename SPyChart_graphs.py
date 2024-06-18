from plotly import graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def plot_spc(df, title):
    """
    Create formatted spc graphs based on the outputs of SPyChart using Plotly Graph Objects

    :param df: results dataframe
    :param title:
    :return: Plotly figure
    """

    rules = ['Rule 1 violation', 'Rule 2 violation', 'Rule 3 violation', 'Rule 4 violation', 'Rule 5 violation']

    cmap = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#17becf']

    # This will create charts for SPC with only one chart.
    if len(df['chart type'].unique()) == 1:

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df['ds'],
                                 y=df['target'],
                                 line=dict(color='#27374D'),
                                 name='Target'))

        fig.add_trace(go.Scatter(x=df['ds'],
                                 y=df['cl'],
                                 line=dict(color='#66a182', width=3, dash='dash'),
                                 name='Central Line'))

        fig.add_trace(go.Scatter(x=df['ds'],
                                 y=df['lcl'],
                                 line=dict(color='#d1495b', width=2, dash='dash'),
                                 name='Lower Control Line'))

        fig.add_trace(go.Scatter(x=df['ds'],
                                 y=df['ucl'],
                                 line=dict(color='#d1495b', width=2, dash='dash'),
                                 name='Upper Control Line'))

        for idx, rule in enumerate(rules):
            df_rule = df[df[rule] == 1]

            fig.add_trace(
                go.Scatter(name=rule,
                           x=df_rule['ds'],
                           y=df_rule['target'],
                           mode='markers',
                           marker=dict(symbol='circle-open',
                                       opacity=1,
                                       color=cmap[idx],
                                       size=12,
                                       line=dict(width=3)
                                       )
                           )
            )

        fig['layout']['xaxis']['title'] = 'date'
        fig['layout']['yaxis']['title'] = 'Process'
        fig['layout']['yaxis'].update(autorange=True)
        fig['layout']['xaxis'].update(autorange=True)

        fig.update_traces(showlegend=False,
                          selector=dict(name='Central Line'))

        fig.update_traces(showlegend=False,
                          selector=dict(name='Lower Control Line'))

        fig.update_traces(showlegend=False,
                          selector=dict(name='Upper Control Line'))

        fig.update_layout(title=title,
                          title_x=.5,
                          yaxis_range=[-100, max(df['target'] * 1.1)],
                          template='plotly_white',
                          margin=dict(l=10, b=10, t=40)
                          )

    else:

        # TODO check whether the below if function is required now that its been concatenated

        if any('XbarS-chart' in type for type in df['chart type'].unique()):
            moving_range = "mS"
        else:
            moving_range = "mR"

        if any('mR-chart' in type for type in df['chart type'].unique()):
            df_bottom = df[df['chart type'] == 'mR-chart']
            df = df[df['chart type'] != 'mR-chart']

        if any('mS-chart' in type for type in df['chart type'].unique()):
            df_bottom = df[df['chart type'] == 'mS-chart']
            df = df[df['chart type'] != 'mS-chart']

        fig = make_subplots(rows=2,
                            cols=1,
                            row_heights=[0.7, 0.3],
                            shared_xaxes=True,
                            vertical_spacing=0.01)

        fig.add_trace(go.Scatter(x=df['ds'],
                                 y=df['target'],
                                 line=dict(color='#27374D'),
                                 name='Target'),
                      row=1,
                      col=1)

        fig.add_trace(go.Scatter(x=df['ds'],
                                 y=df['cl'],
                                 line=dict(color='#66a182', width=3, dash='dash'),
                                 name='Central Line'),
                      row=1, col=1)

        fig.add_trace(go.Scatter(x=df['ds'],
                                 y=df['lcl'],
                                 line=dict(color='#d1495b', width=2, dash='dash'),
                                 name='Lower Control Line'),
                      row=1, col=1)

        fig.add_trace(go.Scatter(x=df['ds'],
                                 y=df['ucl'],
                                 line=dict(color='#d1495b', width=2, dash='dash'),
                                 name='Upper Control Line'),
                      row=1, col=1)

        for idx, rule in enumerate(rules):

            df_rule = df[df[rule] == 1]

            fig.add_trace(
                go.Scatter(name=rule,
                           x=df_rule['ds'],
                           y=df_rule['target'],
                           mode='markers',
                           marker=dict(symbol='circle-open',
                                       opacity=1,
                                       color=cmap[idx],
                                       size=12,
                                       line=dict(width=3))),
                row=1, col=1)

        fig.add_trace(go.Scatter(x=df_bottom['ds'],
                                 y=df_bottom['r'],
                                 line=dict(color="grey"),
                                 name='Moving Range'),
                      row=2, col=1)

        fig.add_trace(go.Scatter(x=df_bottom['ds'],
                                 y=df_bottom['cl'],
                                 line=dict(color='#66a182', width=3, dash='dash'),
                                 name='Central Line'),
                      row=2, col=1)

        fig.add_trace(go.Scatter(x=df_bottom['ds'],
                                 y=df_bottom['lcl'],
                                 line=dict(color='#d1495b', width=2, dash='dash'),
                                 name='Lower Control Line'),
                      row=2, col=1)

        fig.add_trace(go.Scatter(x=df_bottom['ds'],
                                 y=df_bottom['ucl'],
                                 line=dict(color='#d1495b', width=2, dash='dash'),
                                 name='Upper Control Line'),
                      row=2, col=1)

        for idx, rule in enumerate(rules):

            df_rule = df_bottom[df_bottom[rule] == 1]

            fig.add_trace(go.Scatter(name=rule + f' ({moving_range})',
                                     x=df_rule['ds'],
                                     y=df_rule['r'],
                                     mode='markers',
                                     marker=dict(symbol='circle-open',
                                                 opacity=1,
                                                 size=12,
                                                 line=dict(width=3))), row=2, col=1)

        fig['layout']['yaxis'].update(autorange=True)
        fig['layout']['xaxis'].update(autorange=True)

        fig.update_yaxes(title_text="Process", row=1, col=1)
        fig.update_xaxes(title_text="", row=2, col=1)
        fig.update_yaxes(title_text=moving_range, row=2, col=1)

        fig.update_traces(showlegend=False, selector=dict(name='Central Line'), row=2, col=1)
        fig.update_traces(showlegend=False, selector=dict(name='Lower Control Line'), row=2, col=1)
        fig.update_traces(showlegend=False, selector=dict(name='Upper Control Line'), row=2, col=1)

        fig.update_traces(showlegend=False, selector=dict(name='Central Line'), row=1, col=1)
        fig.update_traces(showlegend=False, selector=dict(name='Lower Control Line'), row=1, col=1)
        fig.update_traces(showlegend=False, selector=dict(name='Upper Control Line'), row=1, col=1)

        fig.update_traces(showlegend=False, selector=dict(name='Value'), row=1, col=1)
        fig.update_traces(showlegend=False, selector=dict(name='Moving Range'), row=2, col=1)

        fig.update_layout(title=title,
                          title_x=.5,
                          height=600,
                          template='plotly_white',
                          margin=dict(l=10, b=10, t=40)
                          )

    return fig
