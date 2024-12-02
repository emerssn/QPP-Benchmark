import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union, Optional
import logging
import os
from ..utils.file_utils import ensure_dir

class QPPCorrelationAnalyzer:
    """
    Analyzes correlations between QPP predictions and retrieval effectiveness metrics (nDCG and AP).
    """
    
    def __init__(self, qpp_scores: Dict[str, Dict[str, float]], 
                 retrieval_metrics: Dict[str, Dict[str, Union[float, Dict[str, float]]]],
                 output_dir: Optional[str] = None):
        """
        Initialize the QPP correlation analyzer.
        
        Args:
            qpp_scores: Dictionary of QPP scores {qid: {method: score}}
            retrieval_metrics: Dictionary of retrieval metrics {metric: {per_query: {qid: score}, mean: float}}
                             Only nDCG and AP metrics are supported
            output_dir: Directory to save results and plots
        
        Raises:
            ValueError: If qpp_scores is empty or retrieval_metrics contains no valid metrics
        """
        self.logger = logging.getLogger(__name__)
        
        # Validate input data
        if not qpp_scores:
            raise ValueError("QPP scores dictionary cannot be empty")
        
        # Validate metrics - only allow nDCG and AP
        valid_metrics = {k: v for k, v in retrieval_metrics.items() 
                        if k.lower().startswith('ndcg@') or k.lower() in ['map', 'ap']}
        
        if not valid_metrics:
            raise ValueError("No valid metrics found. Only nDCG@k and AP metrics are supported.")
        
        if len(valid_metrics) != len(retrieval_metrics):
            self.logger.warning("Some metrics were filtered out. Only nDCG@k and AP metrics are supported.")
        
        self.qpp_scores = qpp_scores
        self.retrieval_metrics = valid_metrics
        self.output_dir = ensure_dir(output_dir) if output_dir else None
        
        # Convert data to DataFrames for easier analysis
        self.qpp_df = pd.DataFrame.from_dict(qpp_scores, orient='index')
        self.metrics_df = pd.DataFrame({
            metric: scores['per_query'] 
            for metric, scores in valid_metrics.items()
        })
        
        # Align QIDs between QPP scores and metrics
        self.align_qids()

    def calculate_correlations(self, 
                             correlation_types: List[str] = ['pearson', 'spearman', 'kendall']
                             ) -> Dict[str, pd.DataFrame]:
        """
        Calculate correlations between QPP scores and retrieval metrics.
        
        Args:
            correlation_types: List of correlation coefficients to compute
            
        Returns:
            Dictionary of correlation DataFrames for each correlation type
        """
        self.logger.info("Starting correlation calculations")
        correlations = {}
        
        for corr_type in correlation_types:
            corr_df = pd.DataFrame(index=self.qpp_df.columns, columns=self.metrics_df.columns)
            self.logger.info(f"Calculating {corr_type} correlations")
            
            for qpp_method in self.qpp_df.columns:
                for metric in self.metrics_df.columns:
                    valid_mask = ~(self.qpp_df[qpp_method].isna() | self.metrics_df[metric].isna())
                    x = self.qpp_df[qpp_method][valid_mask]
                    y = self.metrics_df[metric][valid_mask]
                    
                    self.logger.debug(f"Processing QPP Method: {qpp_method}, Metric: {metric}, QIDs used: {len(x)}")
                    
                    try:
                        if len(x) > 1 and len(y) > 1:
                            if corr_type == 'pearson':
                                corr, _ = stats.pearsonr(x, y)
                            elif corr_type == 'spearman':
                                corr, _ = stats.spearmanr(x, y)
                            else:
                                corr, _ = stats.kendalltau(x, y)
                        else:
                            corr = float('nan')
                    except Exception as e:
                        self.logger.warning(f"Error calculating {corr_type} correlation for {qpp_method} vs {metric}: {e}")
                        corr = float('nan')
                    
                    corr_df.loc[qpp_method, metric] = corr
                    
            correlations[corr_type] = corr_df
            self.logger.info(f"Completed {corr_type} correlations")
            
        return correlations

    def plot_correlation_heatmap(self, correlation_type: str = 'kendall', 
                               save_plot: bool = True) -> None:
        """
        Plot heatmap of correlations between QPP methods and retrieval metrics.
        
        Args:
            correlation_type: Type of correlation to plot
            save_plot: Whether to save the plot to file
        """
        correlations = self.calculate_correlations([correlation_type])[correlation_type]
        
        # Convert to numeric, replacing any remaining non-numeric values with NaN
        correlations = correlations.apply(pd.to_numeric, errors='coerce')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlations, annot=True, cmap='RdYlBu', center=0, 
                    vmin=-1, vmax=1, fmt='.3f', mask=correlations.isna())
        plt.title(f'{correlation_type.capitalize()} Correlation between QPP and Retrieval Metrics')
        plt.tight_layout()
        
        if save_plot and self.output_dir:
            plt.savefig(os.path.join(self.output_dir, f'qpp_correlation_{correlation_type}.pdf'))
            plt.close()
        else:
            plt.show()

    def plot_scatter_plots(self, metric: str, save_plots: bool = True) -> None:
        """
        Create scatter plots between each QPP method and a specific retrieval metric.
        
        Args:
            metric: The retrieval metric to plot against
            save_plots: Whether to save the plots to files
        """
        n_methods = len(self.qpp_df.columns)
        fig, axes = plt.subplots(
            (n_methods + 2) // 3, 3,  # Create a 3-column grid
            figsize=(15, 5 * ((n_methods + 2) // 3)),
            squeeze=False
        )
        
        for idx, qpp_method in enumerate(self.qpp_df.columns):
            row, col = idx // 3, idx % 3
            ax = axes[row, col]
            
            sns.regplot(
                x=self.qpp_df[qpp_method],
                y=self.metrics_df[metric],
                ax=ax,
                scatter_kws={'alpha': 0.5}
            )
            
            corr, _ = stats.kendalltau(self.qpp_df[qpp_method], self.metrics_df[metric])
            ax.set_title(f'{qpp_method}\nτ = {corr:.3f}')
            ax.set_xlabel('QPP Score')
            ax.set_ylabel(f'{metric} Score')
            
        # Remove empty subplots
        for idx in range(n_methods, len(axes.flat)):
            fig.delaxes(axes.flat[idx])
            
        plt.tight_layout()
        
        if save_plots and self.output_dir:
            plt.savefig(os.path.join(self.output_dir, f'qpp_scatter_{metric}.pdf'))
            plt.close()
        else:
            plt.show()

    def generate_report(self, correlation_types: List[str] = ['kendall']) -> None:
        """
        Generate a comprehensive correlation analysis report.
        
        Args:
            correlation_types: List of correlation types to include in report
        """
        if not self.output_dir:
            self.logger.warning("No output directory specified. Cannot save report.")
            return
            
        correlations = self.calculate_correlations(correlation_types)
        
        with open(os.path.join(self.output_dir, 'qpp_correlation_report.txt'), 'w') as f:
            f.write("QPP Correlation Analysis Report\n")
            f.write("==============================\n\n")
            f.write("Metrics analyzed: nDCG@k and AP\n\n")
            
            for corr_type, corr_df in correlations.items():
                f.write(f"\n{corr_type.upper()} Correlations:\n")
                f.write("------------------------\n")
                f.write(corr_df.to_string())
                f.write("\n\n")
                
                # Add summary statistics
                f.write("Summary Statistics:\n")
                f.write(f"Best performing QPP method: {corr_df.mean(axis=1).idxmax()}\n")
                f.write(f"Most predictable metric: {corr_df.mean(axis=0).idxmax()}\n")
                f.write("\n")
                
            # Generate plots
            for metric in self.metrics_df.columns:
                self.plot_scatter_plots(metric)
            
            for corr_type in correlation_types:
                self.plot_correlation_heatmap(corr_type)

    def plot_correlations_boxplot(self, correlation_type: str = 'kendall', save_plot: bool = True) -> None:
        """
        Plot boxplot of correlations between QPP methods and retrieval metrics.
        Similar to the original plot_correlations function but adapted for a single dataset.
        
        Args:
            correlation_type: Type of correlation to use ('kendall', 'spearman', or 'pearson')
            save_plot: Whether to save the plot to file
        """
        # Get correlations
        correlations = self.calculate_correlations([correlation_type])[correlation_type]
        
        # Convert to numeric, replacing any remaining non-numeric values with NaN
        correlations = correlations.apply(pd.to_numeric, errors='coerce')
        
        # Create figure
        plt.figure(num=None, figsize=(16, 9), dpi=100, facecolor='w', edgecolor='k')
        
        # Sort methods by median correlation value
        method_medians = correlations.median(axis=1)
        sorted_methods = method_medians.sort_values().index.tolist()
        
        # Prepare data for boxplot
        plot_data = []
        labels = []
        
        for method in sorted_methods:
            data = correlations.loc[method].dropna()
            if len(data) > 0:  # Only include methods with valid data
                plot_data.append(data)
                labels.append(method)
        
        # Create boxplot
        bp = plt.boxplot(plot_data, labels=labels, patch_artist=True)
        
        # Customize boxplot colors
        for box in bp['boxes']:
            box.set(facecolor='lightblue', alpha=0.7)
        
        # Customize plot
        plt.ylabel(f'{correlation_type.capitalize()} Correlation', fontsize=16)
        plt.xlabel('QPP Method', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.title(f'QPP Method Performance across Metrics')
        plt.grid(True, axis='y')
        
        # Add horizontal line at y=0 for reference
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        
        # Set y-axis limits to include all data with some padding
        plt.ylim(min(correlations.min().min() - 0.1, -1), max(correlations.max().max() + 0.1, 1))
        
        # Adjust layout
        plt.tight_layout()
        
        if save_plot and self.output_dir:
            plt.savefig(os.path.join(self.output_dir, f'qpp_correlations_boxplot_{correlation_type}.pdf'))
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_correlations_across_datasets(
        datasets: Dict[str, 'QPPCorrelationAnalyzer'],
        correlation_type: str = 'kendall',
        output_dir: str = None
    ) -> None:
        """
        Plot correlations across multiple datasets.
        
        Args:
            datasets: Dictionary mapping dataset names to their QPPCorrelationAnalyzer instances
            correlation_type: Type of correlation to plot
            output_dir: Directory to save the plot
        """
        # Collect correlations from all datasets
        dataset_correlations = {}
        all_methods = set()
        
        for dataset_name, analyzer in datasets.items():
            correlations = analyzer.calculate_correlations([correlation_type])[correlation_type]
            dataset_correlations[dataset_name] = correlations
            all_methods.update(correlations.index)
        
        # Sort methods by median correlation across all datasets
        all_values = []
        for correlations in dataset_correlations.values():
            all_values.extend(correlations.values.flatten())
        median_by_method = {}
        for method in all_methods:
            method_values = []
            for correlations in dataset_correlations.values():
                if method in correlations.index:
                    method_values.extend(correlations.loc[method].dropna())
            median_by_method[method] = np.nanmedian(method_values)
        
        sorted_methods = sorted(all_methods, key=lambda x: median_by_method[x])
        
        # Prepare data for plotting
        n_methods = len(sorted_methods)
        n_datasets = len(datasets)
        
        # Create figure
        plt.figure(num=None, figsize=(16, 9), dpi=100, facecolor='w', edgecolor='k')
        
        # Position for each dataset's box
        positions = np.arange(n_methods)
        width = 0.8 / n_datasets
        
        # Plot each dataset
        for i, (dataset_name, correlations) in enumerate(dataset_correlations.items()):
            plot_data = []
            for method in sorted_methods:
                if method in correlations.index:
                    plot_data.append(correlations.loc[method].dropna())
                else:
                    plot_data.append([])
            
            # Create boxplot
            bp = plt.boxplot(plot_data,
                            positions=positions + (i - n_datasets/2 + 0.5) * width,
                            widths=width,
                            patch_artist=True,
                            label=dataset_name)
            
            # Set colors
            color = plt.cm.Set3(i / n_datasets)
            for box in bp['boxes']:
                box.set(facecolor=color, alpha=0.7)
        
        # Customize plot
        plt.ylabel(f'{correlation_type.capitalize()} Correlation', fontsize=16)
        plt.xlabel('QPP Method', fontsize=16)
        plt.xticks(positions, sorted_methods, rotation=45, ha='right')
        plt.title(f'QPP Method Performance across Datasets and Metrics')
        plt.grid(True, axis='y')
        
        # Add horizontal line at y=0 for reference
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        
        # Add legend
        plt.legend()
        
        # Set y-axis limits
        all_values = np.array(all_values)
        plt.ylim(min(np.nanmin(all_values) - 0.1, -1), max(np.nanmax(all_values) + 0.1, 1))
        
        # Adjust layout
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'qpp_correlations_across_datasets_{correlation_type}.pdf'))
            plt.close()
        else:
            plt.show()

    def align_qids(self):
        """
        Align QIDs between QPP scores and retrieval metrics.
        """
        # Align QIDs
        common_qids = self.qpp_df.index.intersection(self.metrics_df.index)
        self.qpp_df = self.qpp_df.loc[common_qids]
        self.metrics_df = self.metrics_df.loc[common_qids]
        
        self.logger.info(f"Number of QIDs after alignment: {len(common_qids)}")