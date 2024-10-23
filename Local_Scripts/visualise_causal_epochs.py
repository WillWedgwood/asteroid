import pandas as pd
import plotly.graph_objects as go

def plot_epoch_results(epoch_results_path, results_path):
    # Load the epoch results
    epoch_results_df = pd.read_csv(epoch_results_path)
    results_df = pd.read_csv(results_path)

    # Create a figure
    fig = go.Figure()

    # Add training and validation loss traces
    for lr in epoch_results_df['learning_rate'].unique():
        for wd in epoch_results_df['weight_decay'].unique():
            subset = epoch_results_df[(epoch_results_df['learning_rate'] == lr) & (epoch_results_df['weight_decay'] == wd)]
            fig.add_trace(
                go.Scatter(
                    x=subset['epoch'], y=subset['train_loss'],
                    mode='lines+markers',
                    name=f'Train Loss LR={lr}, WD={wd}',
                    legendgroup=f'LR={lr}, WD={wd}',
                    visible='legendonly'
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=subset['epoch'], y=subset['val_loss'],
                    mode='lines+markers',
                    name=f'Val Loss LR={lr}, WD={wd}',
                    legendgroup=f'LR={lr}, WD={wd}',
                    visible='legendonly'
                )
            )

    # Add table for test scores
    fig.add_trace(
        go.Table(
            header=dict(values=list(results_df.columns)),
            cells=dict(values=[results_df[col] for col in results_df.columns]),
            domain=dict(x=[0, 1], y=[0, 0.2])
        )
    )

    # Update layout
    fig.update_layout(
        height=800,
        title_text="Hyperparameter Tuning Results",
        showlegend=True,
        yaxis=dict(domain=[0.3, 1.0]),  # Adjust the domain of the y-axis for the plots
    )

    # Show the figure
    fig.show()

if __name__ == '__main__':
    # Paths to the CSV files
    epoch_results_path = 'epoch_results.csv'
    results_path = 'hyperparameter_tuning_results.csv'

    # Plot the epoch results
    plot_epoch_results(epoch_results_path, results_path)