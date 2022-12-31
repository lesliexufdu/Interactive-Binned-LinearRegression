import uuid
import hjson
from dash import Dash, dcc, html
import dash
import dash_bootstrap_components as dbc
import dash_auth
from flask_caching import Cache


# configs
with open("./config/config.hjson","r") as f:
    CONFIGS = hjson.load(f)


app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Interactive Binned Linear Regression"
)
# cache
cache = Cache(
    app.server,
    config=CONFIGS["cache_config"]
)
# password of App
auth = dash_auth.BasicAuth(
    app,
    CONFIGS["auth_config"]
)


# basic panel
app.layout = html.Div([
    # unique session-id
    dcc.Store(data=str(uuid.uuid4()), id='session-id'),
    # side bar
    html.Div(
        [
            html.H2("交互式分段逻辑回归", className="display-4"),
            html.Hr(),
            dbc.Nav(
                [
                    dbc.NavLink(
                        [
                            html.Div(page["name"], className="ms-2"),
                        ],
                        href=page["path"],
                        active="exact",
                    )
                    for page in dash.page_registry.values()
                ],
                vertical=True,
                pills=True,
                className="bg-light"
            ),
        ],
        className="side-bar",
    ),
    # content
    html.Div(dash.page_container, className="page-container")
])


if __name__ == '__main__':
    app.run_server(debug=False)