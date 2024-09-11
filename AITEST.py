import pandas as pd
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import dash_table
import io
import base64
import xlsxwriter

# Load and preprocess data
def preprocess_data(excel_file, date_time_col, relevant_columns):
    # Load Excel file
    df = pd.read_excel(excel_file)
    
    # Convert to JSON with only relevant columns
    df = df[relevant_columns]
    df.to_json('defects_data.json', orient='records')
    
    # Load JSON data
    with open('defects_data.json') as f:
        data = json.load(f)
    
    # Convert JSON to DataFrame
    df = pd.DataFrame(data)
    
    return df

# Perform clustering and outlier detection
def analyze_data(df):
    # Encode categorical data
    le = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = le.fit_transform(df[column])
    
    # Clustering to find patterns
    kmeans = KMeans(n_clusters=5)  # Adjust the number of clusters as needed
    df['cluster'] = kmeans.fit_predict(df)
    
    # Outlier detection
    iso_forest = IsolationForest(contamination=0.01)  # Adjust contamination rate as needed
    df['outlier'] = iso_forest.fit_predict(df)
    
    # Extract outliers
    outliers = df[df['outlier'] == -1]
    
    return df, outliers

# Save results to JSON
def save_results(df, outliers):
    # Save results to JSON
    df.to_json('defects_data_with_patterns.json', orient='records')
    
    # Save outliers to JSON
    outliers.to_json('outliers.json', orient='records')

# Create Dash app
def create_dash_app(df, outliers):
    app = dash.Dash(__name__)

    # Create scatter plot
    fig = px.scatter(df, x=df.columns[0], y=df.columns[1], color='cluster', 
                     hover_data=df.columns,
                     title="Cluster Visualization")

    # Convert outliers DataFrame to Excel
    def to_excel():
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            outliers.to_excel(writer, sheet_name='Outliers', index=False)
        output.seek(0)
        return output

    # Define layout
    app.layout = html.Div(style={'backgroundColor': 'black', 'color': 'white', 'padding': '20px'}, children=[
        html.H1("Defect Data Visualization", style={'textAlign': 'center', 'color': 'white'}),
        dcc.Graph(figure=fig),
        html.H2("Outliers", style={'textAlign': 'center', 'color': 'white'}),
        html.Div([
            dcc.Download(
                id='download-dataframe-xlsx'
            ),
            html.Button("Download Outliers as Excel", id="download-btn", n_clicks=0)
        ], style={'textAlign': 'center'}),
        html.Div(id='outlier-list', children=[
            html.Ul([html.Li(f"{row['model']} - {row['part number']} - {row['defect input']}") for _, row in outliers.iterrows()])
        ])
    ])

    @app.callback(
        dash.dependencies.Output("download-dataframe-xlsx", "data"),
        [dash.dependencies.Input("download-btn", "n_clicks")],
        prevent_initial_call=True,
    )
    def download_excel(n_clicks):
        return dcc.send_bytes(to_excel().getvalue(), "outliers.xlsx")

    return app

# Main function
if __name__ == "__main__":
    # Define file paths and column names
    excel_file = 'defects_data.xlsx'
    date_time_col = 'date_time_column_name'  # Replace with the actual column name
    relevant_columns = ['model', 'part number', 'defect input', 'defect symptom', 'defect cause', 'defect location', 'vendor']  # List relevant columns

    # Process and analyze data
    df = preprocess_data(excel_file, date_time_col, relevant_columns)
    df, outliers = analyze_data(df)
    save_results(df, outliers)

    # Create and run Dash app
    app = create_dash_app(df, outliers)
    app.run_server(debug=True)
