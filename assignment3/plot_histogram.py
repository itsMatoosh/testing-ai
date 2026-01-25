"""
Script to create a comparison histogram of Hill Climbing and Random Search results.
Reads from benchmark_results.csv (Hill Climbing) and res_random.csv (Random Search).
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def load_and_expand_data(filepath):
    """
    Load CSV and expand frequency data into individual samples.
    Each row represents (number_of_evaluations, frequency), so we need to
    expand it to have `frequency` number of samples with that evaluation count.
    """
    df = pd.read_csv(filepath)
    
    # Filter out N/A values and convert to numeric
    df = df[df['number_of_evaluations_to_find_crash'] != 'N/A'].copy()
    df['number_of_evaluations_to_find_crash'] = pd.to_numeric(df['number_of_evaluations_to_find_crash'], errors='coerce')
    df['frequency'] = pd.to_numeric(df['frequency'], errors='coerce')
    
    # Drop any rows with NaN values
    df = df.dropna()
    
    # Expand the data: repeat each evaluation count by its frequency
    expanded_data = []
    for _, row in df.iterrows():
        expanded_data.extend([int(row['number_of_evaluations_to_find_crash'])] * int(row['frequency']))
    
    return expanded_data


def create_comparison_histogram(hc_file='benchmark_results.csv', random_file='res_random.csv', 
                                 output_file='comparison_histogram.png'):
    """
    Create a comparison histogram of Hill Climbing and Random Search results.
    
    Args:
        hc_file: Path to Hill Climbing results CSV
        random_file: Path to Random Search results CSV
        output_file: Path to save the output histogram
    """
    # Load data
    hc_data = load_and_expand_data(hc_file)
    random_data = load_and_expand_data(random_file)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Determine bin edges - use the same bins for both datasets for fair comparison
    all_data = hc_data + random_data
    max_val = max(all_data) if all_data else 100
    bin_width = 5
    bins = np.arange(0, max_val + bin_width + 1, bin_width)
    
    # Plot histograms with transparency for overlap visibility
    ax.hist(hc_data, bins=bins, alpha=0.7, label='Hill Climbing', color='#5B9BD5', edgecolor='black')
    ax.hist(random_data, bins=bins, alpha=0.7, label='Random Search', color='#ED7D31', edgecolor='black')
    
    # Labels and title
    ax.set_xlabel('Number of Evaluations to Find Crash')
    ax.set_ylabel('Frequency')
    ax.set_title('Comparison histogram of Hill Climbing and Random Search')
    
    # Legend
    ax.legend(loc='upper right')
    
    # Set integer ticks on y-axis
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Histogram saved to {output_file}")
    
    # Print summary statistics
    print("\n--- Summary Statistics ---")
    print(f"Hill Climbing: n={len(hc_data)}, mean={np.mean(hc_data):.2f}, median={np.median(hc_data):.2f}")
    print(f"Random Search: n={len(random_data)}, mean={np.mean(random_data):.2f}, median={np.median(random_data):.2f}")


def create_accuracy_plot(hc_file='benchmark_results.csv', random_file='res_random.csv',
                          output_file='accuracy_comparison.png'):
    """
    Create a bar plot comparing accuracy (% of runs that found a crash) for each method.
    
    Args:
        hc_file: Path to Hill Climbing results CSV
        random_file: Path to Random Search results CSV
        output_file: Path to save the output plot
    """
    def calculate_accuracy(filepath):
        """Calculate accuracy from a CSV file."""
        df = pd.read_csv(filepath)
        
        # Convert frequency to numeric
        df['frequency'] = pd.to_numeric(df['frequency'], errors='coerce')
        
        # Sum all frequencies
        total_runs = df['frequency'].sum()
        
        # Sum frequencies for N/A (runs that didn't find a crash)
        # Pandas converts "N/A" string to actual NaN, so use pd.isna()
        na_rows = df[pd.isna(df['number_of_evaluations_to_find_crash'])]
        failed_runs = na_rows['frequency'].sum() if len(na_rows) > 0 else 0
        
        # Successful runs
        successful_runs = total_runs - failed_runs
        
        accuracy = (successful_runs / total_runs) * 100 if total_runs > 0 else 0
        
        return successful_runs, total_runs, accuracy
    
    # Calculate accuracy for both methods
    hc_success, hc_total, hc_accuracy = calculate_accuracy(hc_file)
    random_success, random_total, random_accuracy = calculate_accuracy(random_file)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Data for plotting
    methods = ['Hill Climbing', 'Random Search']
    accuracies = [hc_accuracy, random_accuracy]
    colors = ['#5B9BD5', '#ED7D31']
    
    # Create bar plot
    bars = ax.bar(methods, accuracies, color=colors, edgecolor='black', width=0.6)
    
    # Add value labels on bars
    for bar, acc, success, total in zip(bars, accuracies, 
                                         [hc_success, random_success], 
                                         [hc_total, random_total]):
        height = bar.get_height()
        ax.annotate(f'{acc:.1f}%\n({int(success)}/{int(total)})',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Labels and title
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Crash Detection Accuracy: Hill Climbing vs Random Search')
    ax.set_ylim(0, 110)  # Leave room for labels
    
    # Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Accuracy plot saved to {output_file}")
    
    # Print summary
    print("\n--- Accuracy Summary ---")
    print(f"Hill Climbing: {hc_success:.0f}/{hc_total:.0f} runs found a crash ({hc_accuracy:.1f}%)")
    print(f"Random Search: {random_success:.0f}/{random_total:.0f} runs found a crash ({random_accuracy:.1f}%)")


if __name__ == '__main__':
    create_comparison_histogram()
    create_accuracy_plot()
