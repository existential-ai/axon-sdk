import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import logging

class Visualizer:
    def __init__(self, timesteps:list[float], spike_log:dict[str, list[float]], voltage_log:dict[str, list[float]], Vt:float, dt:float):
        self.timesteps = timesteps[0::int(1/dt)]
        self.spike_log = spike_log
        self.voltage_log = self.subsample(voltage_log, dt)
        self.Vt = Vt
        self.app = dash.Dash(__name__)
        self.setup_logging()
        self.setup_layout()
        self.setup_callbacks()

    def subsample(self, voltage_log:dict[str, list[float]], dt:float):
        for key in voltage_log.keys():
            voltage_log[key] = voltage_log[key][0::int(1/dt)]
        return voltage_log

    def setup_logging(self):
        logging.getLogger('werkzeug').setLevel(logging.ERROR)  # Suppress Dash's logging (Werkzeup is the server used by Dash)
        logging.basicConfig(level=logging.CRITICAL)

    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1("Spike dynamics visualization"),
            html.Div([
                dcc.Checklist(
                    id='checklist',
                    options=[{'label': item, 'value': item} for item in self.voltage_log.keys()],
                    value=list(self.voltage_log.keys()),  # Default: all items checked
                    inline=True
                ),
            ]),
            dcc.Graph(id='combined-graph', style={'height': '600px'})
        ])

    def setup_callbacks(self):
        @self.app.callback(
            Output('combined-graph', 'figure'),
            Input('checklist', 'value')
        )
        def update_graph(selected_items):
            selected_items.sort()

            colors = px.colors.qualitative.Plotly
            color_map = {}
            for i, key in enumerate(self.voltage_log.keys()):
                color_map[key] = colors[i % len(colors)]
            
            fig = make_subplots(rows=len(selected_items), cols=1, shared_xaxes=True, vertical_spacing=0)

            for i, item in enumerate(selected_items):
                

                fig.add_trace(go.Scatter(
                        x=self.timesteps,
                        y=self.voltage_log[item],
                        mode='lines+markers',
                        name=item,
                        line=dict(color=color_map[item]),
                    ),
                    row=i + 1,
                    col=1)

                fig.update_yaxes(title_text=item, showgrid=False, showticklabels=False, row=i+1, col=1, range=[-12, 12])

                fig.add_shape(
                    type='line',
                    x0=0,
                    x1=len(self.voltage_log[item])-1,
                    y0=self.Vt,
                    y1=self.Vt,
                    line=dict(color='grey', width=2, dash='dash'),
                    row=i + 1,
                    col=1
                )

                if item in self.spike_log:
                    for spike in self.spike_log[item]:
                        fig.add_shape(
                            type='line',
                            x0=spike,
                            x1=spike,
                            y0=0,
                            y1=12,
                            line=dict(color='gray', width=2, dash='solid'),
                            row=i + 1,
                            col=1
                        )

            fig.update_layout(
                title='Time Series Data',
                template='plotly_white',
                margin=dict(l=20, r=20, t=40, b=20),
                height=150 * len(selected_items),
                showlegend=True,
            )

            return fig

    def run(self):
        self.app.run(debug=False)
