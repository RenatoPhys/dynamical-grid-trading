"""
Backtest vs MT5 - Visualizations
=================================
Gráficos e plots para análise comparativa entre backtest e MT5.

Author: Claude
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from typing import Optional, Tuple
from datetime import datetime


# =============================================================================
# STYLE CONFIGURATION
# =============================================================================
def setup_style():
    """Configura estilo dos gráficos."""
    #plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (14, 8),
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

# Cores consistentes
COLORS = {
    'backtest': '#2196F3',      # Azul
    'mt5': '#FF9800',           # Laranja
    'positive': '#4CAF50',      # Verde
    'negative': '#F44336',      # Vermelho
    'neutral': '#9E9E9E',       # Cinza
    'highlight': '#E91E63',     # Rosa
}


# =============================================================================
# MAIN COMPARISON DASHBOARD
# =============================================================================
def plot_comparison_dashboard(
    matched: pd.DataFrame,
    backtest_full: pd.DataFrame = None,
    mt5_full: pd.DataFrame = None,
    save_path: str = None
) -> plt.Figure:
    """
    Cria dashboard completo de comparação.
    
    Parameters
    ----------
    matched : pd.DataFrame
        DataFrame com trades matcheados
    backtest_full : pd.DataFrame, optional
        DataFrame completo do backtest (para equity curve)
    mt5_full : pd.DataFrame, optional
        DataFrame completo do MT5
    save_path : str, optional
        Caminho para salvar a figura
    
    Returns
    -------
    plt.Figure
        Figura do matplotlib
    """
    setup_style()
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Equity Curves (grande, topo)
    ax1 = fig.add_subplot(gs[0, :])
    _plot_equity_curves(ax1, matched, backtest_full, mt5_full)
    
    # 2. Box-plot diferenças de entrada
    ax2 = fig.add_subplot(gs[1, 0])
    _plot_entry_diff_boxplot(ax2, matched)
    
    # 3. Box-plot diferenças de saída
    ax3 = fig.add_subplot(gs[1, 1])
    _plot_exit_diff_boxplot(ax3, matched)
    
    # 4. Box-plot diferenças de tempo
    ax4 = fig.add_subplot(gs[1, 2])
    _plot_time_diff_boxplot(ax4, matched)
    
    # 5. Scatter: Backtest vs MT5 pontos
    ax5 = fig.add_subplot(gs[2, 0])
    _plot_points_scatter(ax5, matched)
    
    # 6. Histograma de diferenças
    ax6 = fig.add_subplot(gs[2, 1])
    _plot_diff_histogram(ax6, matched)
    
    # 7. Análise por hora do dia
    ax7 = fig.add_subplot(gs[2, 2])
    _plot_hourly_analysis(ax7, matched)
    
    fig.suptitle('Dashboard de Comparação: Backtest vs MetaTrader 5', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"📊 Dashboard salvo em: {save_path}")
    
    return fig


# =============================================================================
# INDIVIDUAL PLOTS
# =============================================================================
def _plot_equity_curves(ax, matched: pd.DataFrame, 
                        backtest_full: pd.DataFrame = None,
                        mt5_full: pd.DataFrame = None):
    """Plota curvas de equity comparativas."""
    
    if len(matched) == 0:
        ax.text(0.5, 0.5, 'Sem dados para plotar', ha='center', va='center')
        ax.set_title('Equity Curves')
        return
    
    # Usar dados dos trades matcheados
    matched_sorted = matched.sort_values('bt_timestamp').copy()
    
    # Calcular equity acumulada
    if 'bt_points' in matched_sorted.columns:
        matched_sorted['bt_cumsum'] = matched_sorted['bt_points'].cumsum()
    
    if 'mt5_profit' in matched_sorted.columns:
        matched_sorted['mt5_cumsum'] = matched_sorted['mt5_profit'].cumsum()
    
    # Plotar
    x = range(len(matched_sorted))
    
    if 'bt_cumsum' in matched_sorted.columns:
        ax.plot(x, matched_sorted['bt_cumsum'], 
                color=COLORS['backtest'], linewidth=2, label='Backtest (pontos)', marker='o', markersize=4)
    
    if 'mt5_cumsum' in matched_sorted.columns:
        ax.plot(x, matched_sorted['mt5_cumsum'], 
                color=COLORS['mt5'], linewidth=2, label='MT5 (R$)', marker='s', markersize=4)
    
    ax.axhline(y=0, color=COLORS['neutral'], linestyle='--', alpha=0.5)
    ax.fill_between(x, 0, matched_sorted.get('bt_cumsum', 0), 
                    alpha=0.1, color=COLORS['backtest'])
    
    ax.set_xlabel('Número do Trade')
    ax.set_ylabel('Resultado Acumulado')
    ax.set_title('📈 Curvas de Equity - Backtest vs MT5')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)


def _plot_entry_diff_boxplot(ax, matched: pd.DataFrame):
    """Box-plot das diferenças de preço de entrada."""
    
    if 'bt_entry_price' not in matched.columns or 'mt5_entry_price' not in matched.columns:
        ax.text(0.5, 0.5, 'Dados insuficientes', ha='center', va='center')
        ax.set_title('Diferença Preço Entrada')
        return
    
    entry_diff = matched['bt_entry_price'] - matched['mt5_entry_price']
    entry_diff = entry_diff.dropna()
    
    if len(entry_diff) == 0:
        ax.text(0.5, 0.5, 'Sem diferenças calculadas', ha='center', va='center')
        ax.set_title('Diferença Preço Entrada')
        return
    
    bp = ax.boxplot(entry_diff, patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor(COLORS['backtest'])
    bp['boxes'][0].set_alpha(0.6)
    bp['medians'][0].set_color(COLORS['highlight'])
    bp['medians'][0].set_linewidth(2)
    
    # Adicionar pontos individuais (jitter)
    x_jitter = np.random.normal(1, 0.04, size=len(entry_diff))
    ax.scatter(x_jitter, entry_diff, alpha=0.4, color=COLORS['backtest'], s=20, zorder=3)
    
    ax.axhline(y=0, color=COLORS['positive'], linestyle='--', alpha=0.7, label='Perfeito')
    
    ax.set_ylabel('Diferença (pontos)')
    ax.set_title('📦 Diferença Preço Entrada\n(Backtest - MT5)')
    ax.set_xticklabels(['Entrada'])
    
    # Estatísticas
    stats_text = f'Média: {entry_diff.mean():.1f}\nStd: {entry_diff.std():.1f}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def _plot_exit_diff_boxplot(ax, matched: pd.DataFrame):
    """Box-plot das diferenças de preço de saída."""
    
    if 'bt_exit_price' not in matched.columns or 'mt5_exit_price' not in matched.columns:
        ax.text(0.5, 0.5, 'Dados insuficientes', ha='center', va='center')
        ax.set_title('Diferença Preço Saída')
        return
    
    exit_diff = matched['bt_exit_price'] - matched['mt5_exit_price']
    exit_diff = exit_diff.dropna()
    
    if len(exit_diff) == 0:
        ax.text(0.5, 0.5, 'Sem diferenças calculadas', ha='center', va='center')
        ax.set_title('Diferença Preço Saída')
        return
    
    bp = ax.boxplot(exit_diff, patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor(COLORS['mt5'])
    bp['boxes'][0].set_alpha(0.6)
    bp['medians'][0].set_color(COLORS['highlight'])
    bp['medians'][0].set_linewidth(2)
    
    # Adicionar pontos individuais
    x_jitter = np.random.normal(1, 0.04, size=len(exit_diff))
    ax.scatter(x_jitter, exit_diff, alpha=0.4, color=COLORS['mt5'], s=20, zorder=3)
    
    ax.axhline(y=0, color=COLORS['positive'], linestyle='--', alpha=0.7)
    
    ax.set_ylabel('Diferença (pontos)')
    ax.set_title('📦 Diferença Preço Saída\n(Backtest - MT5)')
    ax.set_xticklabels(['Saída'])
    
    # Estatísticas
    stats_text = f'Média: {exit_diff.mean():.1f}\nStd: {exit_diff.std():.1f}'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def _plot_time_diff_boxplot(ax, matched: pd.DataFrame):
    """Box-plot das diferenças de tempo."""
    
    if 'bt_timestamp' not in matched.columns or 'mt5_timestamp' not in matched.columns:
        ax.text(0.5, 0.5, 'Dados de tempo\ninsuficientes', ha='center', va='center')
        ax.set_title('Diferença de Tempo')
        return
    
    # Calcular diferença em segundos
    bt_time = pd.to_datetime(matched['bt_timestamp'])
    mt5_time = pd.to_datetime(matched['mt5_timestamp'])
    
    time_diff = (bt_time - mt5_time).dt.total_seconds()
    time_diff = time_diff.dropna()
    
    if len(time_diff) == 0:
        ax.text(0.5, 0.5, 'Sem diferenças calculadas', ha='center', va='center')
        ax.set_title('Diferença de Tempo')
        return
    
    bp = ax.boxplot(time_diff, patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor(COLORS['positive'])
    bp['boxes'][0].set_alpha(0.6)
    bp['medians'][0].set_color(COLORS['highlight'])
    bp['medians'][0].set_linewidth(2)
    
    # Adicionar pontos individuais
    x_jitter = np.random.normal(1, 0.04, size=len(time_diff))
    ax.scatter(x_jitter, time_diff, alpha=0.4, color=COLORS['positive'], s=20, zorder=3)
    
    ax.axhline(y=0, color=COLORS['neutral'], linestyle='--', alpha=0.7)
    
    ax.set_ylabel('Diferença (segundos)')
    ax.set_title('⏱️ Diferença de Tempo\n(Backtest - MT5)')
    ax.set_xticklabels(['Tempo'])
    
    # Estatísticas
    stats_text = f'Média: {time_diff.mean():.1f}s\nStd: {time_diff.std():.1f}s'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def _plot_points_scatter(ax, matched: pd.DataFrame):
    """Scatter plot comparando pontos do backtest vs MT5."""
    
    if 'bt_points' not in matched.columns or 'mt5_profit' not in matched.columns:
        ax.text(0.5, 0.5, 'Dados insuficientes', ha='center', va='center')
        ax.set_title('Backtest vs MT5')
        return
    
    bt_points = matched['bt_points'].dropna()
    mt5_profit = matched['mt5_profit'].dropna()
    
    # Alinhar índices
    common_idx = bt_points.index.intersection(mt5_profit.index)
    bt_points = bt_points.loc[common_idx]
    mt5_profit = mt5_profit.loc[common_idx]
    
    if len(bt_points) == 0:
        ax.text(0.5, 0.5, 'Sem dados comuns', ha='center', va='center')
        ax.set_title('Backtest vs MT5')
        return
    
    # Colorir por concordância de sinal
    colors = np.where((bt_points > 0) == (mt5_profit > 0), 
                      COLORS['positive'], COLORS['negative'])
    
    ax.scatter(bt_points, mt5_profit, c=colors, alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    
    # Linha de regressão
    if len(bt_points) > 1:
        z = np.polyfit(bt_points, mt5_profit, 1)
        p = np.poly1d(z)
        x_line = np.linspace(bt_points.min(), bt_points.max(), 100)
        ax.plot(x_line, p(x_line), color=COLORS['highlight'], linestyle='--', 
                alpha=0.7, label=f'Tendência')
    
    # Linha de referência (y=x normalizada)
    min_val = min(bt_points.min(), mt5_profit.min())
    max_val = max(bt_points.max(), mt5_profit.max())
    
    ax.axhline(y=0, color=COLORS['neutral'], linestyle='-', alpha=0.3)
    ax.axvline(x=0, color=COLORS['neutral'], linestyle='-', alpha=0.3)
    
    ax.set_xlabel('Backtest (pontos)')
    ax.set_ylabel('MT5 (R$)')
    ax.set_title('🎯 Correlação: Backtest vs MT5')
    
    # Correlação
    corr = np.corrcoef(bt_points, mt5_profit)[0, 1]
    ax.text(0.05, 0.95, f'Corr: {corr:.3f}', transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


def _plot_diff_histogram(ax, matched: pd.DataFrame):
    """Histograma das diferenças de entrada e saída."""
    
    diffs = []
    labels = []
    colors_hist = []
    
    if 'bt_entry_price' in matched.columns and 'mt5_entry_price' in matched.columns:
        entry_diff = (matched['bt_entry_price'] - matched['mt5_entry_price']).dropna()
        if len(entry_diff) > 0:
            diffs.append(entry_diff)
            labels.append('Entrada')
            colors_hist.append(COLORS['backtest'])
    
    if 'bt_exit_price' in matched.columns and 'mt5_exit_price' in matched.columns:
        exit_diff = (matched['bt_exit_price'] - matched['mt5_exit_price']).dropna()
        if len(exit_diff) > 0:
            diffs.append(exit_diff)
            labels.append('Saída')
            colors_hist.append(COLORS['mt5'])
    
    if not diffs:
        ax.text(0.5, 0.5, 'Dados insuficientes', ha='center', va='center')
        ax.set_title('Distribuição das Diferenças')
        return
    
    # Plotar histogramas sobrepostos
    for diff, label, color in zip(diffs, labels, colors_hist):
        ax.hist(diff, bins=20, alpha=0.5, label=label, color=color, edgecolor='white')
    
    ax.axvline(x=0, color=COLORS['positive'], linestyle='--', linewidth=2, label='Zero')
    
    ax.set_xlabel('Diferença (pontos)')
    ax.set_ylabel('Frequência')
    ax.set_title('📊 Distribuição das Diferenças')
    ax.legend(loc='upper right')


def _plot_hourly_analysis(ax, matched: pd.DataFrame):
    """Análise das diferenças por hora do dia."""
    
    if 'bt_timestamp' not in matched.columns:
        ax.text(0.5, 0.5, 'Sem dados de tempo', ha='center', va='center')
        ax.set_title('Análise por Hora')
        return
    
    matched = matched.copy()
    matched['hour'] = pd.to_datetime(matched['bt_timestamp']).dt.hour
    
    # Calcular diferença média por hora
    if 'bt_entry_price' in matched.columns and 'mt5_entry_price' in matched.columns:
        matched['entry_diff'] = matched['bt_entry_price'] - matched['mt5_entry_price']
        hourly_stats = matched.groupby('hour')['entry_diff'].agg(['mean', 'std', 'count'])
        
        hours = hourly_stats.index
        means = hourly_stats['mean']
        stds = hourly_stats['std'].fillna(0)
        counts = hourly_stats['count']
        
        # Bar plot com erro
        bars = ax.bar(hours, means, yerr=stds, capsize=3, 
                      color=COLORS['backtest'], alpha=0.7, edgecolor='white')
        
        # Colorir barras positivas/negativas
        for bar, mean in zip(bars, means):
            if mean < 0:
                bar.set_color(COLORS['negative'])
                bar.set_alpha(0.7)
        
        ax.axhline(y=0, color=COLORS['neutral'], linestyle='--', alpha=0.7)
        
        ax.set_xlabel('Hora do Dia')
        ax.set_ylabel('Diferença Média (pontos)')
        ax.set_title('🕐 Diferença por Hora\n(com desvio padrão)')
        ax.set_xticks(hours)
        
    else:
        # Se não tem preços, mostrar contagem por hora
        hourly_counts = matched.groupby('hour').size()
        ax.bar(hourly_counts.index, hourly_counts.values, 
               color=COLORS['backtest'], alpha=0.7, edgecolor='white')
        ax.set_xlabel('Hora do Dia')
        ax.set_ylabel('Número de Trades')
        ax.set_title('🕐 Trades por Hora')


# =============================================================================
# DETAILED COMPARISON PLOTS
# =============================================================================
def plot_detailed_boxplots(
    matched: pd.DataFrame,
    save_path: str = None
) -> plt.Figure:
    """
    Box-plots detalhados para análise de diferenças.
    
    Parameters
    ----------
    matched : pd.DataFrame
        DataFrame com trades matcheados
    save_path : str, optional
        Caminho para salvar
    
    Returns
    -------
    plt.Figure
    """
    setup_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Calcular diferenças
    diffs = {}
    
    if 'bt_entry_price' in matched.columns and 'mt5_entry_price' in matched.columns:
        diffs['Entrada'] = (matched['bt_entry_price'] - matched['mt5_entry_price']).dropna()
    
    if 'bt_exit_price' in matched.columns and 'mt5_exit_price' in matched.columns:
        diffs['Saída'] = (matched['bt_exit_price'] - matched['mt5_exit_price']).dropna()
    
    if 'bt_timestamp' in matched.columns and 'mt5_timestamp' in matched.columns:
        time_diff = (pd.to_datetime(matched['bt_timestamp']) - 
                     pd.to_datetime(matched['mt5_timestamp'])).dt.total_seconds()
        diffs['Tempo (s)'] = time_diff.dropna()
    
    if 'bt_points' in matched.columns and 'mt5_profit' in matched.columns:
        # Normalizar para comparar (assumindo point_value = 0.20)
        diffs['Resultado'] = (matched['bt_points'] * 0.20 - matched['mt5_profit']).dropna()
    
    # Plot 1: Box-plots lado a lado
    ax1 = axes[0, 0]
    if diffs:
        data_to_plot = [diffs[k].values for k in diffs.keys() if len(diffs[k]) > 0]
        labels = [k for k in diffs.keys() if len(diffs[k]) > 0]
        
        bp = ax1.boxplot(data_to_plot, patch_artist=True, labels=labels)
        colors_bp = [COLORS['backtest'], COLORS['mt5'], COLORS['positive'], COLORS['highlight']]
        for patch, color in zip(bp['boxes'], colors_bp[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    
    ax1.axhline(y=0, color=COLORS['neutral'], linestyle='--', alpha=0.7)
    ax1.set_ylabel('Diferença')
    ax1.set_title('📦 Box-plots de Todas as Diferenças')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Violin plot de entrada
    ax2 = axes[0, 1]
    if 'Entrada' in diffs and len(diffs['Entrada']) > 0:
        vp = ax2.violinplot(diffs['Entrada'], positions=[1], showmeans=True, showmedians=True)
        vp['bodies'][0].set_facecolor(COLORS['backtest'])
        vp['bodies'][0].set_alpha(0.6)
        
        # Adicionar pontos
        x_jitter = np.random.normal(1, 0.05, size=len(diffs['Entrada']))
        ax2.scatter(x_jitter, diffs['Entrada'], alpha=0.3, color=COLORS['backtest'], s=15)
    
    ax2.axhline(y=0, color=COLORS['positive'], linestyle='--', alpha=0.7)
    ax2.set_ylabel('Diferença (pontos)')
    ax2.set_title('🎻 Violin Plot - Diferença de Entrada')
    ax2.set_xticks([1])
    ax2.set_xticklabels(['Entrada'])
    
    # Plot 3: Violin plot de saída
    ax3 = axes[1, 0]
    if 'Saída' in diffs and len(diffs['Saída']) > 0:
        vp = ax3.violinplot(diffs['Saída'], positions=[1], showmeans=True, showmedians=True)
        vp['bodies'][0].set_facecolor(COLORS['mt5'])
        vp['bodies'][0].set_alpha(0.6)
        
        x_jitter = np.random.normal(1, 0.05, size=len(diffs['Saída']))
        ax3.scatter(x_jitter, diffs['Saída'], alpha=0.3, color=COLORS['mt5'], s=15)
    
    ax3.axhline(y=0, color=COLORS['positive'], linestyle='--', alpha=0.7)
    ax3.set_ylabel('Diferença (pontos)')
    ax3.set_title('🎻 Violin Plot - Diferença de Saída')
    ax3.set_xticks([1])
    ax3.set_xticklabels(['Saída'])
    
    # Plot 4: Q-Q plot ou distribuição cumulativa
    ax4 = axes[1, 1]
    if 'Entrada' in diffs and len(diffs['Entrada']) > 0:
        sorted_data = np.sort(diffs['Entrada'])
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax4.plot(sorted_data, cdf, color=COLORS['backtest'], linewidth=2, label='Entrada')
    
    if 'Saída' in diffs and len(diffs['Saída']) > 0:
        sorted_data = np.sort(diffs['Saída'])
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax4.plot(sorted_data, cdf, color=COLORS['mt5'], linewidth=2, label='Saída')
    
    ax4.axvline(x=0, color=COLORS['positive'], linestyle='--', alpha=0.7)
    ax4.set_xlabel('Diferença (pontos)')
    ax4.set_ylabel('CDF')
    ax4.set_title('📈 Distribuição Cumulativa das Diferenças')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle('Análise Detalhada das Diferenças: Backtest vs MT5', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"📊 Gráfico salvo em: {save_path}")
    
    return fig


def plot_time_series_comparison(
    matched: pd.DataFrame,
    save_path: str = None
) -> plt.Figure:
    """
    Comparação temporal das diferenças ao longo do tempo.
    
    Parameters
    ----------
    matched : pd.DataFrame
        DataFrame com trades matcheados
    save_path : str, optional
        Caminho para salvar
    
    Returns
    -------
    plt.Figure
    """
    setup_style()
    
    if 'bt_timestamp' not in matched.columns:
        print("Sem dados de timestamp para plotar série temporal")
        return None
    
    matched = matched.copy()
    matched['timestamp'] = pd.to_datetime(matched['bt_timestamp'])
    matched = matched.sort_values('timestamp')
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Diferença de preço de entrada ao longo do tempo
    ax1 = axes[0]
    if 'bt_entry_price' in matched.columns and 'mt5_entry_price' in matched.columns:
        entry_diff = matched['bt_entry_price'] - matched['mt5_entry_price']
        
        colors = np.where(entry_diff >= 0, COLORS['positive'], COLORS['negative'])
        ax1.bar(range(len(entry_diff)), entry_diff, color=colors, alpha=0.7, width=0.8)
        ax1.axhline(y=0, color=COLORS['neutral'], linestyle='-', alpha=0.5)
        
        # Média móvel
        if len(entry_diff) > 5:
            rolling_mean = entry_diff.rolling(window=5, min_periods=1).mean()
            ax1.plot(range(len(rolling_mean)), rolling_mean, 
                     color=COLORS['highlight'], linewidth=2, label='Média Móvel (5)')
            ax1.legend(loc='upper right')
    
    ax1.set_ylabel('Diferença Entrada (pts)')
    ax1.set_title('📊 Diferença de Preço de Entrada ao Longo do Tempo')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Diferença de preço de saída ao longo do tempo
    ax2 = axes[1]
    if 'bt_exit_price' in matched.columns and 'mt5_exit_price' in matched.columns:
        exit_diff = matched['bt_exit_price'] - matched['mt5_exit_price']
        
        colors = np.where(exit_diff >= 0, COLORS['positive'], COLORS['negative'])
        ax2.bar(range(len(exit_diff)), exit_diff, color=colors, alpha=0.7, width=0.8)
        ax2.axhline(y=0, color=COLORS['neutral'], linestyle='-', alpha=0.5)
        
        if len(exit_diff) > 5:
            rolling_mean = exit_diff.rolling(window=5, min_periods=1).mean()
            ax2.plot(range(len(rolling_mean)), rolling_mean, 
                     color=COLORS['highlight'], linewidth=2, label='Média Móvel (5)')
            ax2.legend(loc='upper right')
    
    ax2.set_ylabel('Diferença Saída (pts)')
    ax2.set_title('📊 Diferença de Preço de Saída ao Longo do Tempo')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Diferença acumulada
    ax3 = axes[2]
    if 'bt_points' in matched.columns:
        bt_cumsum = matched['bt_points'].cumsum()
        ax3.plot(range(len(bt_cumsum)), bt_cumsum, 
                 color=COLORS['backtest'], linewidth=2, label='Backtest')
    
    if 'mt5_profit' in matched.columns:
        mt5_cumsum = matched['mt5_profit'].cumsum()
        ax3.plot(range(len(mt5_cumsum)), mt5_cumsum, 
                 color=COLORS['mt5'], linewidth=2, label='MT5')
    
    ax3.axhline(y=0, color=COLORS['neutral'], linestyle='--', alpha=0.5)
    ax3.set_xlabel('Número do Trade')
    ax3.set_ylabel('Resultado Acumulado')
    ax3.set_title('📈 Resultado Acumulado: Backtest vs MT5')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    fig.suptitle('Análise Temporal: Backtest vs MT5', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"📊 Gráfico salvo em: {save_path}")
    
    return fig


def plot_daily_comparison(
    matched: pd.DataFrame,
    save_path: str = None
) -> plt.Figure:
    """
    Comparação diária entre backtest e MT5.
    
    Parameters
    ----------
    matched : pd.DataFrame
        DataFrame com trades matcheados
    save_path : str, optional
        Caminho para salvar
    
    Returns
    -------
    plt.Figure
    """
    setup_style()
    
    if 'bt_timestamp' not in matched.columns:
        print("Sem dados de timestamp")
        return None
    
    matched = matched.copy()
    matched['date'] = pd.to_datetime(matched['bt_timestamp']).dt.date
    
    # Agregar por dia
    daily = matched.groupby('date').agg({
        'bt_points': 'sum',
        'mt5_profit': 'sum',
        'bt_entry_price': 'count'  # contagem de trades
    }).rename(columns={'bt_entry_price': 'trade_count'})
    
    daily['bt_points'] = daily['bt_points'].fillna(0)
    daily['mt5_profit'] = daily['mt5_profit'].fillna(0)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    x = range(len(daily))
    width = 0.35
    
    # Plot 1: Resultado diário lado a lado
    ax1 = axes[0]
    bars1 = ax1.bar([i - width/2 for i in x], daily['bt_points'], width, 
                     label='Backtest (pts)', color=COLORS['backtest'], alpha=0.7)
    bars2 = ax1.bar([i + width/2 for i in x], daily['mt5_profit'], width, 
                     label='MT5 (R$)', color=COLORS['mt5'], alpha=0.7)
    
    ax1.axhline(y=0, color=COLORS['neutral'], linestyle='-', alpha=0.5)
    ax1.set_ylabel('Resultado')
    ax1.set_title('📅 Resultado Diário: Backtest vs MT5')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Diferença diária
    ax2 = axes[1]
    # Normalizar backtest para R$ (assumindo point_value = 0.20)
    daily['diff'] = daily['bt_points'] * 0.20 - daily['mt5_profit']
    
    colors = np.where(daily['diff'] >= 0, COLORS['positive'], COLORS['negative'])
    ax2.bar(x, daily['diff'], color=colors, alpha=0.7)
    ax2.axhline(y=0, color=COLORS['neutral'], linestyle='-', alpha=0.5)
    
    ax2.set_xlabel('Data')
    ax2.set_ylabel('Diferença (R$)')
    ax2.set_title('📊 Diferença Diária (Backtest - MT5)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(d) for d in daily.index], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Análise Diária: Backtest vs MT5', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"📊 Gráfico salvo em: {save_path}")
    
    return fig


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================
def generate_all_plots(
    matched: pd.DataFrame,
    output_dir: str = '.',
    prefix: str = 'comparison'
) -> None:
    """
    Gera todos os gráficos de comparação.
    
    Parameters
    ----------
    matched : pd.DataFrame
        DataFrame com trades matcheados
    output_dir : str
        Diretório para salvar os gráficos
    prefix : str
        Prefixo para os nomes dos arquivos
    """
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Gerando gráficos de comparação...")
    print("=" * 50)
    
    # Dashboard principal
    plot_comparison_dashboard(
        matched, 
        save_path=str(output_path / f'{prefix}_dashboard.png')
    )
    
    # Box-plots detalhados
    plot_detailed_boxplots(
        matched,
        save_path=str(output_path / f'{prefix}_boxplots.png')
    )
    
    # Série temporal
    plot_time_series_comparison(
        matched,
        save_path=str(output_path / f'{prefix}_timeseries.png')
    )
    
    # Comparação diária
    plot_daily_comparison(
        matched,
        save_path=str(output_path / f'{prefix}_daily.png')
    )
    
    print("=" * 50)
    print(f"✅ Todos os gráficos salvos em: {output_dir}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    """
    Demo com dados simulados.
    """
    print("=" * 70)
    print("DEMO: Gerando gráficos com dados simulados")
    print("=" * 70)
    
    # Criar dados de exemplo
    np.random.seed(42)
    n_trades = 50
    
    base_timestamps = pd.date_range('2024-02-01 09:00', periods=n_trades, freq='15min')
    
    demo_matched = pd.DataFrame({
        'bt_timestamp': base_timestamps,
        'mt5_timestamp': base_timestamps + pd.Timedelta(seconds=np.random.randint(-30, 30)),
        'bt_direction': np.random.choice(['BUY', 'SELL'], n_trades),
        'mt5_direction': np.random.choice(['BUY', 'SELL'], n_trades),
        'bt_entry_price': 127500 + np.random.randn(n_trades) * 100,
        'bt_exit_price': 127500 + np.random.randn(n_trades) * 100,
        'bt_points': np.random.choice([-30, -20, 20, 50, 100], n_trades, p=[0.2, 0.15, 0.25, 0.25, 0.15]),
        'mt5_profit': np.random.choice([-6, -4, 4, 10, 20], n_trades, p=[0.2, 0.15, 0.25, 0.25, 0.15]),
    })
    
    # Adicionar pequenas diferenças realistas
    demo_matched['mt5_entry_price'] = demo_matched['bt_entry_price'] + np.random.randn(n_trades) * 3
    demo_matched['mt5_exit_price'] = demo_matched['bt_exit_price'] + np.random.randn(n_trades) * 5
    
    # Gerar todos os gráficos
    generate_all_plots(demo_matched, output_dir='.', prefix='demo_comparison')
    
    # Mostrar um gráfico
    plt.show()
    
    print("\n" + "=" * 70)
    print("Para usar com seus dados reais:")
    print("  from compare_visualizations import generate_all_plots")
    print("  generate_all_plots(matched_trades, output_dir='./reports')")
    print("=" * 70)
