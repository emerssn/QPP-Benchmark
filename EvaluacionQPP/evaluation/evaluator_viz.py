import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import os
from typing import Dict, Optional
from ..utils.file_utils import ensure_dir

class RetrievalMetricsVisualizer:
    """
    Visualizes retrieval evaluation metrics with various plots.
    Generates boxplots, histograms, mean metric bars, and scatter plots.
    """
    
    def __init__(self, eval_results: Dict, output_dir: Optional[str] = None):
        self.eval_results = eval_results
        self.output_dir = output_dir
        self.metrics_df = self._prepare_metrics_df()
        
    def _prepare_metrics_df(self) -> pd.DataFrame:
        """Convert per-query metrics into a DataFrame."""
        data = {}
        for metric, metric_data in self.eval_results.items():
            data[metric] = metric_data['per_query']
        return pd.DataFrame(data)
    
    def _format_metric_name(self, metric: str) -> str:
        """Format metric names for display in plots."""
        if metric.startswith('ndcg@'):
            k = metric.split('@')[1]
            return f'nDCG@{k}'
        elif metric == 'ap':
            return 'AP'
        elif metric.startswith('p@'):
            k = metric.split('@')[1]
            return f'P@{k}'
        elif metric.startswith('rr@'):
            k = metric.split('@')[1]
            return f'RR@{k}'
        else:
            return metric
    
    def _save_plot(self, filename: str):
        """Helper to save plots consistently."""
        if self.output_dir:
            ensure_dir(self.output_dir)
            path = os.path.join(self.output_dir, filename)
            plt.savefig(f'{path}.png', dpi=300, bbox_inches='tight')
            plt.savefig(f'{path}.pdf', bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_metric_distributions(self, save: bool = True):
        """Generate boxplots and histograms showing metric distributions."""
        # Boxplot
        plt.figure(figsize=(12, 8))
        df_formatted = self.metrics_df.rename(columns=self._format_metric_name)
        sns.boxplot(data=df_formatted, palette='viridis')
        plt.title('Distribución de las Métricas por Consulta')
        plt.ylabel('Puntuación')
        plt.xticks(rotation=45)
        if save:
            self._save_plot('boxplot_metricas')
        else:
            plt.show()
        
        # Histograms
        n_metrics = len(self.metrics_df.columns)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        for i, metric in enumerate(self.metrics_df.columns):
            ax = axes[i]
            formatted_name = self._format_metric_name(metric)
            sns.histplot(self.metrics_df[metric], kde=True, ax=ax, color='skyblue')
            ax.set_title(formatted_name)
            ax.set_xlabel('Puntuación')
            ax.set_ylabel('Frecuencia')
        
        # Hide empty subplots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        if save:
            self._save_plot('histogramas_metricas')
        else:
            plt.show()

    def plot_mean_metrics(self, save: bool = True):
        """Bar plot showing mean values for each metric."""
        means = {self._format_metric_name(metric): data['mean'] 
                for metric, data in self.eval_results.items()}
        metrics = list(means.keys())
        values = list(means.values())
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=metrics, y=values, palette='rocket')
        plt.title('Media de las Métricas de Evaluación')
        plt.ylabel('Media')
        plt.xticks(rotation=45)
        if save:
            self._save_plot('media_metricas')
        else:
            plt.show()

    def plot_metric_relationships(self, save: bool = True):
        """Scatter plots showing relationships between metrics."""
        metrics = self.metrics_df.columns.tolist()
        for i in range(len(metrics)):
            for j in range(i+1, len(metrics)):
                metric1 = metrics[i]
                metric2 = metrics[j]
                x = self.metrics_df[metric1]
                y = self.metrics_df[metric2]
                
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=x, y=y, alpha=0.6, color='#3498db')
                plt.title(f'Relación entre {self._format_metric_name(metric1)} y {self._format_metric_name(metric2)}')
                plt.xlabel(self._format_metric_name(metric1))
                plt.ylabel(self._format_metric_name(metric2))
                
                # Add correlation coefficient
                corr = stats.pearsonr(x, y)[0]
                plt.annotate(f'r = {corr:.2f}', 
                            xy=(0.05, 0.95), xycoords='axes fraction',
                            fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
                
                if save:
                    filename = f'scatter_{metric1}_vs_{metric2}'
                    self._save_plot(filename)
                else:
                    plt.show()

    def generate_all_plots(self, save: bool = True):
        """Generate all available visualizations."""
        self.plot_metric_distributions(save)
        self.plot_mean_metrics(save)
        self.plot_metric_relationships(save)