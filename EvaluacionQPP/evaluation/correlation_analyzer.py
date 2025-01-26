import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Union, Optional
import logging
import os
from ..utils.file_utils import ensure_dir

METHOD_NAME_MAP = {
    'idf_avg': 'IDF Promedio',
    'idf_max': 'IDF Máximo',
    'scq_avg': 'SCQ Promedio',
    'scq_max': 'SCQ Máximo',
    'wig': 'WIG',
    'nqc': 'NQC',
    'clarity': 'Clarity',
    'uef_wig': 'UEF-WIG',
    'uef_nqc': 'UEF-NQC'
}

class QPPCorrelationAnalyzer:
    """
    Analyzes correlations between QPP predictions and retrieval effectiveness metrics (nDCG and AP).
    """
    
    def __init__(self, qpp_scores: Dict[str, Dict[str, float]], 
                 retrieval_metrics: Dict[str, Dict[str, Union[float, Dict[str, float]]]],
                 output_dir: Optional[str] = None,
                 dpi: int = 300):
        """
        Initialize the QPP correlation analyzer.
        
        Args:
            qpp_scores: Dictionary of QPP scores {qid: {method: score}}
            retrieval_metrics: Dictionary of retrieval metrics {metric: {per_query: {qid: score}, mean: float}}
                             Only nDCG and AP metrics are supported
            output_dir: Directory to save results and plots
            dpi: DPI (dots per inch) for saved plots
        
        Raises:
            ValueError: If qpp_scores is empty or retrieval_metrics contains no valid metrics
        """
        self.logger = logging.getLogger(__name__)
        self.dpi = dpi
        

        # Validate input data
        if not qpp_scores:
            raise ValueError("QPP scores dictionary cannot be empty")
        
        # Validate metrics - only allow nDCG and AP
        valid_metrics = {k: v for k, v in retrieval_metrics.items() 
                        if any(k.lower().startswith(prefix) 
                              for prefix in ['ndcg@', 'map', 'ap', 'p@', 'rr@'])}
        
        if not valid_metrics:
            raise ValueError("No valid metrics found. Supported metrics: nDCG@k, AP/MAP, P@k, RR@k")
            
        if len(valid_metrics) != len(retrieval_metrics):
            self.logger.warning("Some metrics were filtered out. Only nDCG@k and AP metrics are supported.")
        
        # Add validation for QPP scores
        for method, scores in qpp_scores.items():
            if not all(isinstance(score, (int, float)) for score in scores.values()):
                raise ValueError(f"Non-numeric QPP scores found in method: {method}")
                
        self.qpp_scores = qpp_scores
        self.retrieval_metrics = valid_metrics
        self.output_dir = ensure_dir(output_dir) if output_dir else None
        # Align query sets between QPP and metrics
        common_qids = set(qpp_scores.keys()) & set(retrieval_metrics['ndcg@10']['per_query'].keys())
        
        self.qpp_scores = {k:v for k,v in qpp_scores.items() if k in common_qids}
        self.retrieval_metrics = {
            metric: {
                'per_query': {k:v for k,v in scores['per_query'].items() if k in common_qids},
                'mean': np.mean(list(scores['per_query'].values()))
            } 
            for metric, scores in retrieval_metrics.items()
        }
        
        # Convert data to DataFrames for easier analysis
        self.qpp_df = pd.DataFrame.from_dict(qpp_scores, orient='index')
        self.metrics_df = pd.DataFrame({
            metric: scores['per_query'] 
            for metric, scores in valid_metrics.items()
        })
        
        # Align QIDs between QPP scores and metrics
        self.align_qids()

    def calculate_correlations(self, 
                             correlation_types: List[str] = ['pearson', 'spearman', 'kendall'],
                             min_queries: int = 5,
                             min_results_per_query: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Calculate correlations between QPP scores and retrieval metrics.
        
        Args:
            correlation_types: List of correlation coefficients to compute
            min_queries: Minimum number of queries required for correlation calculation
            min_results_per_query: Minimum number of results required per query
        """
        correlations = {}
        
        for corr_type in correlation_types:
            corr_df = pd.DataFrame(index=self.qpp_df.columns, columns=self.metrics_df.columns)
            
            for qpp_method in self.qpp_df.columns:
                for metric in self.metrics_df.columns:
                    # Filter out queries with insufficient results if threshold provided
                    valid_mask = ~(self.qpp_df[qpp_method].isna() | self.metrics_df[metric].isna())
                    
                    x = self.qpp_df[qpp_method][valid_mask]
                    y = self.metrics_df[metric][valid_mask]
                    
                    try:
                        if len(x) >= min_queries:
                            if corr_type == 'pearson':
                                corr, p_value = stats.pearsonr(x, y)
                            elif corr_type == 'spearman':
                                corr, p_value = stats.spearmanr(x, y)
                            else:  # kendall
                                corr, p_value = stats.kendalltau(x, y)
                                
                            corr_df.loc[qpp_method, metric] = corr
                            
                            # Log if not significant
                            if p_value >= 0.05:
                                self.logger.info(
                                    f"Non-significant correlation for {qpp_method} vs {metric} "
                                    f"(p={p_value:.3f}, n={len(x)})"
                                )
                        else:
                            corr_df.loc[qpp_method, metric] = float('nan')
                            self.logger.warning(
                                f"Insufficient queries for {qpp_method} vs {metric} "
                                f"(n={len(x)} < {min_queries})"
                            )
                    except Exception as e:
                        self.logger.warning(f"Error calculating {corr_type} correlation: {e}")
                        corr_df.loc[qpp_method, metric] = float('nan')
            
            correlations[corr_type] = corr_df
            
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
        
        # Convert to numeric, replacing any non-numeric values with NaN
        correlations = correlations.apply(pd.to_numeric, errors='coerce')
        
        # Map method names to Spanish
        correlations.index = correlations.index.map(lambda x: METHOD_NAME_MAP.get(x, x))
        
        plt.figure(figsize=(10, 8))
        # Create mask for NaN values
        mask = np.isnan(correlations)
        
        sns.heatmap(correlations, annot=True, cmap='RdYlBu', center=0, 
                    vmin=-1, vmax=1, fmt='.3f', mask=mask)
        plt.title(f'Correlación {correlation_type.capitalize()} entre QPP y Métricas de Recuperación')
        plt.tight_layout()
        
        if save_plot and self.output_dir:
            plt.savefig(os.path.join(self.output_dir, f'correlacion_qpp_{correlation_type}.pdf'), dpi=self.dpi)
            plt.savefig(os.path.join(self.output_dir, f'correlacion_qpp_{correlation_type}.png'), dpi=self.dpi)
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
            method_name = METHOD_NAME_MAP.get(qpp_method, qpp_method)
            ax.set_title(f'{method_name}\nτ = {corr:.3f}')
            ax.set_xlabel('Puntuación QPP')
            ax.set_ylabel(f'Puntuación {metric}')
            
        # Remove empty subplots
        for idx in range(n_methods, len(axes.flat)):
            fig.delaxes(axes.flat[idx])
            
        plt.tight_layout()
        
        if save_plots and self.output_dir:
            plt.savefig(os.path.join(self.output_dir, f'dispersion_qpp_{metric}.pdf'), dpi=self.dpi)
            plt.savefig(os.path.join(self.output_dir, f'dispersion_qpp_{metric}.png'), dpi=self.dpi)
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
            self.logger.warning("No se especificó directorio de salida. No se puede guardar el informe.")
            return
            
        correlations = self.calculate_correlations(correlation_types)
        
        with open(os.path.join(self.output_dir, 'informe_correlacion_qpp.txt'), 'w') as f:
            f.write("Informe de Análisis de Correlación QPP\n")
            f.write("=====================================\n\n")
            f.write("Métricas analizadas: nDCG@k y AP\n\n")
            
            for corr_type, corr_df in correlations.items():
                f.write(f"\nCorrelaciones {corr_type.upper()}:\n")
                f.write("------------------------\n")
                
                f.write("\nNúmero de consultas utilizadas:\n")
                for method in corr_df.index:
                    n_queries = len(self.qpp_df[method].dropna())
                    method_name = METHOD_NAME_MAP.get(method, method)
                    f.write(f"{method_name}: {n_queries} consultas\n")
                
                f.write("\nValores de correlación (solo estadísticamente significativos):\n")
                # Map method names for display
                display_df = corr_df.copy()
                display_df.index = display_df.index.map(lambda x: METHOD_NAME_MAP.get(x, x))
                f.write(display_df.to_string())
                f.write("\n\n")
                
                f.write("Estadísticas Resumen:\n")
                best_method = corr_df.mean(axis=1).idxmax()
                best_metric = corr_df.mean(axis=0).idxmax()
                f.write(f"Mejor método QPP: {METHOD_NAME_MAP.get(best_method, best_method)}\n")
                f.write(f"Métrica más predecible: {best_metric}\n")
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
        
        # Map method names to Spanish
        labels = [METHOD_NAME_MAP.get(method, method) for method in labels]
        
        # Create boxplot
        bp = plt.boxplot(plot_data, labels=labels, patch_artist=True)
        
        # Customize boxplot colors
        for box in bp['boxes']:
            box.set(facecolor='lightblue', alpha=0.7)
        
        # Customize plot
        plt.ylabel(f'Correlación {correlation_type.capitalize()}', fontsize=16)
        plt.xlabel('Método QPP', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Rendimiento de Métodos QPP a través de Métricas')
        plt.grid(True, axis='y')
        
        # Add horizontal line at y=0 for reference
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        
        # Set y-axis limits to include all data with some padding
        plt.ylim(min(correlations.min().min() - 0.1, -1), max(correlations.max().max() + 0.1, 1))
        
        # Adjust layout
        plt.tight_layout()
        
        if save_plot and self.output_dir:
            plt.savefig(os.path.join(self.output_dir, f'correlaciones_qpp_boxplot_{correlation_type}.pdf'), dpi=self.dpi)
            plt.savefig(os.path.join(self.output_dir, f'correlaciones_qpp_boxplot_{correlation_type}.png'), dpi=self.dpi)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def plot_correlations_across_datasets(
        datasets: Dict[str, 'QPPCorrelationAnalyzer'],
        correlation_type: str = 'kendall',
        output_dir: str = None,
        dpi: int = 300
    ) -> None:
        """
        Plot correlations across multiple datasets.
        
        Args:
            datasets: Dictionary mapping dataset names to their QPPCorrelationAnalyzer instances
            correlation_type: Type of correlation to plot
            output_dir: Directory to save the plot
            dpi: DPI for saved plots
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
            plt.savefig(os.path.join(output_dir, f'qpp_correlations_across_datasets_{correlation_type}.pdf'), dpi=dpi)
            plt.savefig(os.path.join(output_dir, f'qpp_correlations_across_datasets_{correlation_type}.png'), dpi=dpi)
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