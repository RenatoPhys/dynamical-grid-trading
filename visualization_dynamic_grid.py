"""
Visualization Module for Dynamic Grid Trading Backtest
=======================================================
Provides comprehensive visualizations for analyzing trading backtest results.

Author: Renato Critelli
"""

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# VISUALIZATION CONFIG
# =============================================================================
@dataclass
class PlotConfig:
    """Configuration for plot styling."""
    figsize: Tuple[int, int] = (12, 6)
    style: str = 'seaborn-v0_8-whitegrid'
    title_fontsize: int = 14
    label_fontsize: int = 12
    grid_alpha: float = 0.3
    line_width: float = 1.5
    colors: dict = None
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                'equity': '#2E86AB',
                'drawdown': '#E94F37',
                'buy': '#28A745',
                'sell': '#DC3545',
                'total': '#1E3A5F',
                'neutral': '#6C757D'
            }


# =============================================================================
# BACKTEST VISUALIZER CLASS
# =============================================================================
class BacktestVisualizer:
    """
    Visualization class for dynamic grid trading backtest results.
    
    Parameters
    ----------
    results : pd.DataFrame
        Full results DataFrame from run_backtest()
    trades : pd.DataFrame
        Trades-only DataFrame from run_backtest()
    symbol : str, optional
        Trading symbol name (default: 'WIN')
    timeframe : str, optional
        Timeframe description (default: 'Tick')
    config : PlotConfig, optional
        Plot configuration settings
    
    Example
    -------
    >>> results, trades = run_backtest(tick_data, config)
    >>> viz = BacktestVisualizer(results, trades, symbol='WIN', timeframe='Tick')
    >>> viz.plot_equity_curve()
    >>> plt.show()
    """
    
    def __init__(
        self,
        results: pd.DataFrame,
        trades: pd.DataFrame,
        symbol: str = 'WIN',
        timeframe: str = 'Tick',
        config: Optional[PlotConfig] = None
    ):
        self.results = results.copy()
        self.trades = trades.copy()
        self.symbol = symbol
        self.timeframe = timeframe
        self.config = config or PlotConfig()
        
        # Prepare data for visualization
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare and enrich data for visualization."""
        # Ensure timestamp is datetime
        if 'timestamp' in self.results.columns:
            self.results['timestamp'] = pd.to_datetime(self.results['timestamp'])
            self.results = self.results.set_index('timestamp')
        
        # Calculate drawdown
        self.results['equity'] = self.results['accumulated_profit']
        self.results['peak'] = self.results['equity'].cummax()
        self.results['drawdown'] = self.results['equity'] - self.results['peak']
        self.results['drawdown_pct'] = np.where(
            self.results['peak'] != 0,
            self.results['drawdown'] / self.results['peak'],
            0
        )
        
        # Add position column (1 = buy, -1 = sell, 0 = neutral)
        self.results['position'] = 0
        self.results.loc[self.results['flag_buy'] == 1, 'position'] = 1
        self.results.loc[self.results['flag_sell'] == 1, 'position'] = -1
        
        # Prepare trades DataFrame
        if 'timestamp' in self.trades.columns:
            self.trades['timestamp'] = pd.to_datetime(self.trades['timestamp'])
            self.trades = self.trades.set_index('timestamp')
        
        self.trades['position'] = 0
        self.trades.loc[self.trades['flag_buy'] == 1, 'position'] = 1
        self.trades.loc[self.trades['flag_sell'] == 1, 'position'] = -1
        
        # Create daily resampled data
        self._create_daily_data()
    
    def _create_daily_data(self):
        """Create daily resampled DataFrame for daily visualizations."""
        trades = self.trades.copy()
        
        # Daily aggregation from trades
        self.daily = trades.groupby(trades.index.date).agg({
            'profit': 'sum',
            'trade_points': ['sum', 'count'],
            'trade_status': lambda x: (x == 1).sum(),  # TP count
        }).reset_index()
        
        self.daily.columns = ['date', 'profit', 'points', 'num_trades', 'tp_count']
        self.daily['date'] = pd.to_datetime(self.daily['date'])
        self.daily = self.daily.set_index('date')
        
        # Add SL and EOD counts
        sl_counts = trades[trades['trade_status'] == -1].groupby(trades[trades['trade_status'] == -1].index.date).size()
        eod_counts = trades[trades['trade_status'] == 0].groupby(trades[trades['trade_status'] == 0].index.date).size()
        
        self.daily['sl_count'] = self.daily.index.map(lambda x: sl_counts.get(x.date(), 0))
        self.daily['eod_count'] = self.daily.index.map(lambda x: eod_counts.get(x.date(), 0))
        
        # Buy/Sell breakdown
        buy_profit = trades[trades['position'] == 1].groupby(trades[trades['position'] == 1].index.date)['profit'].sum()
        sell_profit = trades[trades['position'] == -1].groupby(trades[trades['position'] == -1].index.date)['profit'].sum()
        buy_count = trades[trades['position'] == 1].groupby(trades[trades['position'] == 1].index.date).size()
        sell_count = trades[trades['position'] == -1].groupby(trades[trades['position'] == -1].index.date).size()
        
        self.daily['profit_buy'] = self.daily.index.map(lambda x: buy_profit.get(x.date(), 0))
        self.daily['profit_sell'] = self.daily.index.map(lambda x: sell_profit.get(x.date(), 0))
        self.daily['num_buys'] = self.daily.index.map(lambda x: buy_count.get(x.date(), 0))
        self.daily['num_sells'] = self.daily.index.map(lambda x: sell_count.get(x.date(), 0))
        
        # Cumulative metrics
        self.daily['cum_profit'] = self.daily['profit'].cumsum()
        self.daily['cum_profit_buy'] = self.daily['profit_buy'].cumsum()
        self.daily['cum_profit_sell'] = self.daily['profit_sell'].cumsum()
        self.daily['cum_trades'] = self.daily['num_trades'].cumsum()
        
        # Drawdown (daily)
        self.daily['peak'] = self.daily['cum_profit'].cummax()
        self.daily['drawdown'] = self.daily['cum_profit'] - self.daily['peak']
        self.daily['drawdown_pct'] = np.where(
            self.daily['peak'] != 0,
            self.daily['drawdown'] / self.daily['peak'],
            0
        )
        
        # Win rate per day
        wins = trades[trades['profit'] > 0].groupby(trades[trades['profit'] > 0].index.date).size()
        self.daily['wins'] = self.daily.index.map(lambda x: wins.get(x.date(), 0))
        self.daily['win_rate'] = np.where(
            self.daily['num_trades'] > 0,
            self.daily['wins'] / self.daily['num_trades'] * 100,
            0
        )
        
        # Rolling metrics (7-day and 30-day)
        self.daily['profit_ma7'] = self.daily['profit'].rolling(7, min_periods=1).mean()
        self.daily['profit_ma30'] = self.daily['profit'].rolling(30, min_periods=1).mean()
        self.daily['win_rate_ma7'] = self.daily['win_rate'].rolling(7, min_periods=1).mean()
    
    def get_daily_data(self) -> pd.DataFrame:
        """
        Return the daily resampled DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Daily aggregated data with columns:
            - profit, points, num_trades
            - tp_count, sl_count, eod_count
            - profit_buy, profit_sell, num_buys, num_sells
            - cum_profit, cum_profit_buy, cum_profit_sell
            - drawdown, drawdown_pct
            - wins, win_rate
            - profit_ma7, profit_ma30, win_rate_ma7
        """
        return self.daily.copy()

    # =========================================================================
    # EQUITY CURVE
    # =========================================================================
    def plot_equity_curve(
        self,
        figsize: Optional[Tuple[int, int]] = None,
        include_drawdown: bool = True,
        title_suffix: str = ''
    ):
        """
        Plot the equity curve with optional drawdown subplot.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure dimensions (width, height)
        include_drawdown : bool
            If True, includes a drawdown subplot (default: True)
        title_suffix : str
            Additional text to append to the title
        
        Returns
        -------
        matplotlib.pyplot
            The pyplot object for further customization
        """
        figsize = figsize or self.config.figsize
        colors = self.config.colors
        
        # Build title
        base_title = f'Curva de Equity - {self.symbol} ({self.timeframe})'
        if title_suffix:
            base_title += f'\n{title_suffix}'
        
        if include_drawdown:
            fig, (ax1, ax2) = plt.subplots(
                2, 1, 
                figsize=(figsize[0], figsize[1] * 1.5),
                gridspec_kw={'height_ratios': [3, 1]}, 
                sharex=True
            )
            
            # Equity curve
            ax1.plot(
                self.results.index, 
                self.results['equity'],
                color=colors['equity'],
                linewidth=self.config.line_width
            )
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax1.set_title(base_title, fontsize=self.config.title_fontsize)
            ax1.set_ylabel('Resultado (R$)', fontsize=self.config.label_fontsize)
            ax1.grid(True, alpha=self.config.grid_alpha)
            
            # Fill positive/negative regions
            ax1.fill_between(
                self.results.index,
                self.results['equity'],
                0,
                where=self.results['equity'] >= 0,
                color=colors['buy'],
                alpha=0.2
            )
            ax1.fill_between(
                self.results.index,
                self.results['equity'],
                0,
                where=self.results['equity'] < 0,
                color=colors['sell'],
                alpha=0.2
            )
            
            # Drawdown subplot
            ax2.fill_between(
                self.results.index, 
                self.results['drawdown'], 
                0, 
                color=colors['drawdown'], 
                alpha=0.3
            )
            ax2.plot(
                self.results.index,
                self.results['drawdown'],
                color=colors['drawdown'],
                linewidth=0.8
            )
            ax2.set_title('Drawdown (R$)', fontsize=self.config.title_fontsize - 2)
            ax2.set_ylabel('Drawdown (R$)', fontsize=self.config.label_fontsize)
            ax2.set_xlabel('Data/Hora', fontsize=self.config.label_fontsize)
            ax2.grid(True, alpha=self.config.grid_alpha)
            
            plt.tight_layout()
        else:
            plt.figure(figsize=figsize)
            plt.plot(
                self.results.index, 
                self.results['equity'],
                color=colors['equity'],
                linewidth=self.config.line_width
            )
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            plt.title(base_title, fontsize=self.config.title_fontsize)
            plt.xlabel('Data/Hora', fontsize=self.config.label_fontsize)
            plt.ylabel('Resultado (R$)', fontsize=self.config.label_fontsize)
            plt.grid(True, alpha=self.config.grid_alpha)
            plt.tight_layout()
        
        return plt

    # =========================================================================
    # DRAWDOWN ANALYSIS
    # =========================================================================
    def plot_drawdown(self, figsize: Optional[Tuple[int, int]] = None):
        """
        Plot detailed drawdown analysis.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure dimensions (width, height)
        
        Returns
        -------
        matplotlib.pyplot
        """
        figsize = figsize or self.config.figsize
        colors = self.config.colors
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1] * 1.3), sharex=True)
        
        # Drawdown in R$
        ax1.fill_between(
            self.results.index, 
            self.results['drawdown'], 
            0, 
            color=colors['drawdown'], 
            alpha=0.3
        )
        ax1.plot(
            self.results.index,
            self.results['drawdown'],
            color=colors['drawdown'],
            linewidth=0.8
        )
        ax1.set_title(
            f'Análise de Drawdown - {self.symbol} ({self.timeframe})',
            fontsize=self.config.title_fontsize
        )
        ax1.set_ylabel('Drawdown (R$)', fontsize=self.config.label_fontsize)
        ax1.grid(True, alpha=self.config.grid_alpha)
        
        # Drawdown percentage
        ax2.fill_between(
            self.results.index,
            self.results['drawdown_pct'] * 100,
            0,
            color=colors['drawdown'],
            alpha=0.3
        )
        ax2.plot(
            self.results.index,
            self.results['drawdown_pct'] * 100,
            color=colors['drawdown'],
            linewidth=0.8
        )
        ax2.set_ylabel('Drawdown (%)', fontsize=self.config.label_fontsize)
        ax2.set_xlabel('Data/Hora', fontsize=self.config.label_fontsize)
        ax2.grid(True, alpha=self.config.grid_alpha)
        
        plt.tight_layout()
        return plt

    # =========================================================================
    # BY POSITION (BUY/SELL)
    # =========================================================================
    def plot_by_position(self, figsize: Optional[Tuple[int, int]] = None):
        """
        Plot equity curve separated by buy and sell positions.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure dimensions (width, height)
        
        Returns
        -------
        matplotlib.pyplot
        """
        figsize = figsize or self.config.figsize
        colors = self.config.colors
        
        # Calculate cumulative profits by position
        df = self.results.copy()
        df['profit_buy'] = np.where(df['flag_buy'] == 1, df['profit'], 0)
        df['profit_sell'] = np.where(df['flag_sell'] == 1, df['profit'], 0)
        df['cum_buy'] = df['profit_buy'].cumsum()
        df['cum_sell'] = df['profit_sell'].cumsum()
        
        plt.figure(figsize=figsize)
        plt.plot(
            df.index, df['equity'], 
            label='Total', 
            color=colors['total'],
            linewidth=2
        )
        plt.plot(
            df.index, df['cum_buy'], 
            label='Compras', 
            color=colors['buy'],
            linewidth=1.5
        )
        plt.plot(
            df.index, df['cum_sell'], 
            label='Vendas', 
            color=colors['sell'],
            linewidth=1.5
        )
        
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.title(
            f'Retorno Acumulado por Tipo de Posição - {self.symbol} ({self.timeframe})',
            fontsize=self.config.title_fontsize
        )
        plt.xlabel('Data/Hora', fontsize=self.config.label_fontsize)
        plt.ylabel('Resultado (R$)', fontsize=self.config.label_fontsize)
        plt.grid(True, alpha=self.config.grid_alpha)
        plt.legend()
        plt.tight_layout()
        
        return plt

    # =========================================================================
    # PROFIT BY HOUR
    # =========================================================================
    def plot_profit_by_hour(self, figsize: Optional[Tuple[int, int]] = None):
        """
        Plot total profit by hour of day with buy/sell breakdown.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure dimensions (width, height)
        
        Returns
        -------
        matplotlib.pyplot
        """
        figsize = figsize or (14, 8)
        colors = self.config.colors
        
        # Prepare data
        trades = self.trades.copy()
        trades['hour'] = trades.index.hour
        
        # Aggregate by hour
        hourly = trades.groupby('hour').agg({
            'profit': 'sum',
            'trade_points': 'count'
        }).rename(columns={'trade_points': 'num_trades'})
        
        # Separate by position
        hourly_buy = trades[trades['position'] == 1].groupby('hour')['profit'].sum()
        hourly_sell = trades[trades['position'] == -1].groupby('hour')['profit'].sum()
        
        # Ensure all hours are represented
        all_hours = range(24)
        hourly = hourly.reindex(all_hours, fill_value=0)
        hourly_buy = hourly_buy.reindex(all_hours, fill_value=0)
        hourly_sell = hourly_sell.reindex(all_hours, fill_value=0)
        
        # Filter to hours with trades
        active_hours = hourly[hourly['num_trades'] > 0].index.tolist()
        if not active_hours:
            print("Nenhum trade encontrado para plotar.")
            return plt
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        bar_width = 0.25
        x = np.arange(len(active_hours))
        
        bars1 = ax.bar(
            x - bar_width, 
            hourly.loc[active_hours, 'profit'], 
            width=bar_width, 
            label='Total', 
            color=colors['total'], 
            alpha=0.8
        )
        bars2 = ax.bar(
            x, 
            hourly_buy.loc[active_hours], 
            width=bar_width, 
            label='Compras', 
            color=colors['buy'], 
            alpha=0.8
        )
        bars3 = ax.bar(
            x + bar_width, 
            hourly_sell.loc[active_hours], 
            width=bar_width, 
            label='Vendas', 
            color=colors['sell'], 
            alpha=0.8
        )
        
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.set_title(
            f'Lucro Total por Hora do Dia - {self.symbol} ({self.timeframe})',
            fontsize=self.config.title_fontsize
        )
        ax.set_xlabel('Hora do Dia', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Resultado (R$)', fontsize=self.config.label_fontsize)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{h:02d}h' for h in active_hours])
        ax.grid(True, axis='y', alpha=self.config.grid_alpha)
        ax.legend()
        
        plt.tight_layout()
        return plt

    # =========================================================================
    # CUMULATIVE BY HOUR
    # =========================================================================
    def plot_cumulative_by_hour(self, figsize: Optional[Tuple[int, int]] = None):
        """
        Plot cumulative profit evolution for each hour of the day.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure dimensions (width, height)
        
        Returns
        -------
        matplotlib.pyplot
        """
        figsize = figsize or (14, 10)
        
        # Prepare data
        df = self.results.copy()
        df['hour'] = df.index.hour
        df['date'] = df.index.date
        
        # Get unique hours with trades
        trades_hours = df[(df['flag_buy'] == 1) | (df['flag_sell'] == 1)]['hour'].unique()
        hours = sorted(trades_hours)
        
        if len(hours) == 0:
            print("Nenhum trade encontrado para plotar.")
            return plt
        
        # Create daily aggregation
        daily_data = {}
        for hour in hours:
            hour_mask = df['hour'] == hour
            buy_mask = hour_mask & (df['flag_buy'] == 1)
            sell_mask = hour_mask & (df['flag_sell'] == 1)
            
            df[f'profit_h{hour}'] = np.where(hour_mask, df['profit'], 0)
            df[f'profit_h{hour}_buy'] = np.where(buy_mask, df['profit'], 0)
            df[f'profit_h{hour}_sell'] = np.where(sell_mask, df['profit'], 0)
        
        # Aggregate by date
        daily_df = df.groupby('date').agg({
            **{f'profit_h{h}': 'sum' for h in hours},
            **{f'profit_h{h}_buy': 'sum' for h in hours},
            **{f'profit_h{h}_sell': 'sum' for h in hours}
        })
        daily_df.index = pd.to_datetime(daily_df.index)
        
        # Calculate cumulative sums
        for hour in hours:
            daily_df[f'cum_h{hour}'] = daily_df[f'profit_h{hour}'].cumsum()
            daily_df[f'cum_h{hour}_buy'] = daily_df[f'profit_h{hour}_buy'].cumsum()
            daily_df[f'cum_h{hour}_sell'] = daily_df[f'profit_h{hour}_sell'].cumsum()
        
        # Create color palette
        cmap = cm.get_cmap('tab20', len(hours))
        
        # Create subplots
        fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        # Total
        for i, hour in enumerate(hours):
            axs[0].plot(
                daily_df.index, 
                daily_df[f'cum_h{hour}'], 
                label=f'{hour:02d}h', 
                color=cmap(i),
                linewidth=1.2
            )
        axs[0].set_title(
            f'Lucro Acumulado por Hora do Dia - {self.symbol} ({self.timeframe})',
            fontsize=self.config.title_fontsize
        )
        axs[0].set_ylabel('Resultado (R$)', fontsize=self.config.label_fontsize)
        axs[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axs[0].grid(True, alpha=self.config.grid_alpha)
        axs[0].legend(loc='upper left', bbox_to_anchor=(1.01, 1), ncol=2)
        
        # Buys only
        for i, hour in enumerate(hours):
            axs[1].plot(
                daily_df.index, 
                daily_df[f'cum_h{hour}_buy'], 
                label=f'{hour:02d}h', 
                color=cmap(i),
                linewidth=1.2
            )
        axs[1].set_title('Lucro Acumulado por Hora - Apenas Compras', fontsize=self.config.title_fontsize - 2)
        axs[1].set_ylabel('Resultado (R$)', fontsize=self.config.label_fontsize)
        axs[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axs[1].grid(True, alpha=self.config.grid_alpha)
        axs[1].legend(loc='upper left', bbox_to_anchor=(1.01, 1), ncol=2)
        
        # Sells only
        for i, hour in enumerate(hours):
            axs[2].plot(
                daily_df.index, 
                daily_df[f'cum_h{hour}_sell'], 
                label=f'{hour:02d}h', 
                color=cmap(i),
                linewidth=1.2
            )
        axs[2].set_title('Lucro Acumulado por Hora - Apenas Vendas', fontsize=self.config.title_fontsize - 2)
        axs[2].set_xlabel('Data', fontsize=self.config.label_fontsize)
        axs[2].set_ylabel('Resultado (R$)', fontsize=self.config.label_fontsize)
        axs[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axs[2].grid(True, alpha=self.config.grid_alpha)
        axs[2].legend(loc='upper left', bbox_to_anchor=(1.01, 1), ncol=2)
        
        fig.suptitle('Análise de Lucro Acumulado por Hora', fontsize=16, y=1.02)
        plt.tight_layout()
        
        return plt

    # =========================================================================
    # TRADE DISTRIBUTION
    # =========================================================================
    def plot_trade_distribution(self, figsize: Optional[Tuple[int, int]] = None):
        """
        Plot distribution of trade results (histogram).
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure dimensions (width, height)
        
        Returns
        -------
        matplotlib.pyplot
        """
        figsize = figsize or self.config.figsize
        colors = self.config.colors
        
        fig, axes = plt.subplots(1, 2, figsize=(figsize[0], figsize[1]))
        
        # Histogram of profits
        axes[0].hist(
            self.trades['profit'], 
            bins=30, 
            color=colors['equity'], 
            alpha=0.7,
            edgecolor='white'
        )
        axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        axes[0].axvline(
            x=self.trades['profit'].mean(), 
            color=colors['sell'], 
            linestyle='-', 
            linewidth=2,
            label=f'Média: R$ {self.trades["profit"].mean():.2f}'
        )
        axes[0].set_title('Distribuição dos Resultados', fontsize=self.config.title_fontsize)
        axes[0].set_xlabel('Resultado (R$)', fontsize=self.config.label_fontsize)
        axes[0].set_ylabel('Frequência', fontsize=self.config.label_fontsize)
        axes[0].legend()
        axes[0].grid(True, alpha=self.config.grid_alpha)
        
        # Histogram of points
        axes[1].hist(
            self.trades['trade_points'], 
            bins=30, 
            color=colors['total'], 
            alpha=0.7,
            edgecolor='white'
        )
        axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        axes[1].axvline(
            x=self.trades['trade_points'].mean(), 
            color=colors['sell'], 
            linestyle='-', 
            linewidth=2,
            label=f'Média: {self.trades["trade_points"].mean():.1f} pts'
        )
        axes[1].set_title('Distribuição dos Pontos', fontsize=self.config.title_fontsize)
        axes[1].set_xlabel('Pontos', fontsize=self.config.label_fontsize)
        axes[1].set_ylabel('Frequência', fontsize=self.config.label_fontsize)
        axes[1].legend()
        axes[1].grid(True, alpha=self.config.grid_alpha)
        
        plt.suptitle(
            f'Análise de Distribuição - {self.symbol} ({self.timeframe})',
            fontsize=self.config.title_fontsize + 2
        )
        plt.tight_layout()
        
        return plt

    # =========================================================================
    # TRADE OUTCOMES PIE CHART
    # =========================================================================
    def plot_trade_outcomes(self, figsize: Optional[Tuple[int, int]] = None):
        """
        Plot pie chart of trade outcomes (TP, SL, EOD).
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure dimensions (width, height)
        
        Returns
        -------
        matplotlib.pyplot
        """
        figsize = figsize or (10, 5)
        colors = self.config.colors
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Trade outcomes (TP, SL, EOD)
        outcomes = self.trades['trade_status'].value_counts()
        labels_map = {1: 'Take Profit', -1: 'Stop Loss', 0: 'Fim do Dia'}
        labels = [labels_map.get(k, str(k)) for k in outcomes.index]
        colors_pie = [colors['buy'], colors['sell'], colors['neutral']]
        
        axes[0].pie(
            outcomes.values, 
            labels=labels, 
            autopct='%1.1f%%',
            colors=colors_pie[:len(outcomes)],
            startangle=90
        )
        axes[0].set_title('Resultados dos Trades', fontsize=self.config.title_fontsize)
        
        # Win/Loss ratio
        wins = len(self.trades[self.trades['profit'] > 0])
        losses = len(self.trades[self.trades['profit'] < 0])
        breakeven = len(self.trades[self.trades['profit'] == 0])
        
        sizes = [wins, losses, breakeven]
        labels = [f'Ganhos ({wins})', f'Perdas ({losses})', f'Zero ({breakeven})']
        colors_wl = [colors['buy'], colors['sell'], colors['neutral']]
        
        axes[1].pie(
            sizes, 
            labels=labels, 
            autopct='%1.1f%%',
            colors=colors_wl,
            startangle=90
        )
        axes[1].set_title('Win/Loss Ratio', fontsize=self.config.title_fontsize)
        
        plt.suptitle(
            f'Análise de Trades - {self.symbol} ({self.timeframe})',
            fontsize=self.config.title_fontsize + 2
        )
        plt.tight_layout()
        
        return plt

    # =========================================================================
    # DAILY PERFORMANCE
    # =========================================================================
    def plot_daily_performance(self, figsize: Optional[Tuple[int, int]] = None):
        """
        Plot daily performance bar chart.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure dimensions (width, height)
        
        Returns
        -------
        matplotlib.pyplot
        """
        figsize = figsize or (14, 6)
        colors = self.config.colors
        
        # Aggregate by day
        trades = self.trades.copy()
        trades['date'] = trades.index.date
        
        daily = trades.groupby('date').agg({
            'profit': 'sum',
            'trade_points': ['sum', 'count']
        })
        daily.columns = ['profit', 'points', 'num_trades']
        daily['cum_profit'] = daily['profit'].cumsum()
        
        fig, axes = plt.subplots(2, 1, figsize=(figsize[0], figsize[1] * 1.5), sharex=True)
        
        # Daily profit bars
        bar_colors = [colors['buy'] if x >= 0 else colors['sell'] for x in daily['profit']]
        axes[0].bar(range(len(daily)), daily['profit'], color=bar_colors, alpha=0.8)
        axes[0].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        axes[0].set_title(
            f'Resultado Diário - {self.symbol} ({self.timeframe})',
            fontsize=self.config.title_fontsize
        )
        axes[0].set_ylabel('Resultado (R$)', fontsize=self.config.label_fontsize)
        axes[0].grid(True, axis='y', alpha=self.config.grid_alpha)
        
        # Add number of trades as text
        for i, (profit, num) in enumerate(zip(daily['profit'], daily['num_trades'])):
            axes[0].annotate(
                f'{num}', 
                xy=(i, profit), 
                ha='center', 
                va='bottom' if profit >= 0 else 'top',
                fontsize=8,
                alpha=0.7
            )
        
        # Cumulative profit line
        axes[1].plot(
            range(len(daily)), 
            daily['cum_profit'],
            color=colors['equity'],
            linewidth=2,
            marker='o',
            markersize=4
        )
        axes[1].fill_between(
            range(len(daily)),
            daily['cum_profit'],
            0,
            where=daily['cum_profit'] >= 0,
            color=colors['buy'],
            alpha=0.2
        )
        axes[1].fill_between(
            range(len(daily)),
            daily['cum_profit'],
            0,
            where=daily['cum_profit'] < 0,
            color=colors['sell'],
            alpha=0.2
        )
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1].set_title('Lucro Acumulado', fontsize=self.config.title_fontsize)
        axes[1].set_xlabel('Dia', fontsize=self.config.label_fontsize)
        axes[1].set_ylabel('Resultado Acumulado (R$)', fontsize=self.config.label_fontsize)
        axes[1].grid(True, alpha=self.config.grid_alpha)
        
        # X-axis labels
        axes[1].set_xticks(range(len(daily)))
        axes[1].set_xticklabels(
            [d.strftime('%d/%m') for d in daily.index], 
            rotation=45, 
            ha='right'
        )
        
        plt.tight_layout()
        return plt

    # =========================================================================
    # COMPLETE DASHBOARD
    # =========================================================================
    def plot_dashboard(self, figsize: Optional[Tuple[int, int]] = None):
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure dimensions (width, height)
        
        Returns
        -------
        matplotlib.pyplot
        """
        figsize = figsize or (16, 12)
        colors = self.config.colors
        
        fig = plt.figure(figsize=figsize)
        
        # Define grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Equity curve (top, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(
            self.results.index, 
            self.results['equity'],
            color=colors['equity'],
            linewidth=1.5
        )
        ax1.fill_between(
            self.results.index,
            self.results['equity'],
            0,
            where=self.results['equity'] >= 0,
            color=colors['buy'],
            alpha=0.2
        )
        ax1.fill_between(
            self.results.index,
            self.results['equity'],
            0,
            where=self.results['equity'] < 0,
            color=colors['sell'],
            alpha=0.2
        )
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title('Curva de Equity', fontsize=12)
        ax1.set_ylabel('Resultado (R$)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Trade outcomes pie (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        wins = len(self.trades[self.trades['profit'] > 0])
        losses = len(self.trades[self.trades['profit'] < 0])
        sizes = [wins, losses]
        ax2.pie(
            sizes, 
            labels=[f'Ganhos\n({wins})', f'Perdas\n({losses})'],
            colors=[colors['buy'], colors['sell']],
            autopct='%1.1f%%',
            startangle=90
        )
        ax2.set_title('Win/Loss', fontsize=12)
        
        # 3. Drawdown (middle left, spans 2 columns)
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.fill_between(
            self.results.index, 
            self.results['drawdown'], 
            0, 
            color=colors['drawdown'], 
            alpha=0.3
        )
        ax3.set_title('Drawdown', fontsize=12)
        ax3.set_ylabel('Drawdown (R$)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Distribution histogram (middle right)
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.hist(
            self.trades['profit'], 
            bins=20, 
            color=colors['equity'], 
            alpha=0.7,
            edgecolor='white'
        )
        ax4.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        ax4.set_title('Distribuição', fontsize=12)
        ax4.set_xlabel('Resultado (R$)')
        
        # 5. By position (bottom, spans full width)
        ax5 = fig.add_subplot(gs[2, :])
        df = self.results.copy()
        df['cum_buy'] = np.where(df['flag_buy'] == 1, df['profit'], 0).cumsum()
        df['cum_sell'] = np.where(df['flag_sell'] == 1, df['profit'], 0).cumsum()
        
        ax5.plot(df.index, df['equity'], label='Total', color=colors['total'], linewidth=2)
        ax5.plot(df.index, df['cum_buy'], label='Compras', color=colors['buy'], linewidth=1.5)
        ax5.plot(df.index, df['cum_sell'], label='Vendas', color=colors['sell'], linewidth=1.5)
        ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax5.set_title('Resultado por Tipo de Posição', fontsize=12)
        ax5.set_xlabel('Data/Hora')
        ax5.set_ylabel('Resultado (R$)')
        ax5.legend(loc='upper left')
        ax5.grid(True, alpha=0.3)
        
        fig.suptitle(
            f'Dashboard de Backtest - {self.symbol} ({self.timeframe})',
            fontsize=16,
            fontweight='bold'
        )
        
        plt.tight_layout()
        return plt

    # =========================================================================
    # DAILY VISUALIZATION METHODS
    # =========================================================================
    def plot_daily_equity(self, figsize: Optional[Tuple[int, int]] = None, include_drawdown: bool = True):
        """
        Plot daily equity curve with optional drawdown.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure dimensions (width, height)
        include_drawdown : bool
            If True, includes a drawdown subplot (default: True)
        
        Returns
        -------
        matplotlib.pyplot
        """
        figsize = figsize or self.config.figsize
        colors = self.config.colors
        
        if include_drawdown:
            fig, (ax1, ax2) = plt.subplots(
                2, 1,
                figsize=(figsize[0], figsize[1] * 1.5),
                gridspec_kw={'height_ratios': [3, 1]},
                sharex=True
            )
            
            # Daily equity curve
            ax1.plot(
                self.daily.index,
                self.daily['cum_profit'],
                color=colors['equity'],
                linewidth=2,
                marker='o',
                markersize=4
            )
            ax1.fill_between(
                self.daily.index,
                self.daily['cum_profit'],
                0,
                where=self.daily['cum_profit'] >= 0,
                color=colors['buy'],
                alpha=0.2
            )
            ax1.fill_between(
                self.daily.index,
                self.daily['cum_profit'],
                0,
                where=self.daily['cum_profit'] < 0,
                color=colors['sell'],
                alpha=0.2
            )
            ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax1.set_title(
                f'Curva de Equity Diária - {self.symbol} ({self.timeframe})',
                fontsize=self.config.title_fontsize
            )
            ax1.set_ylabel('Resultado Acumulado (R$)', fontsize=self.config.label_fontsize)
            ax1.grid(True, alpha=self.config.grid_alpha)
            
            # Daily drawdown
            ax2.fill_between(
                self.daily.index,
                self.daily['drawdown'],
                0,
                color=colors['drawdown'],
                alpha=0.3
            )
            ax2.plot(
                self.daily.index,
                self.daily['drawdown'],
                color=colors['drawdown'],
                linewidth=1
            )
            ax2.set_title('Drawdown Diário (R$)', fontsize=self.config.title_fontsize - 2)
            ax2.set_ylabel('Drawdown (R$)', fontsize=self.config.label_fontsize)
            ax2.set_xlabel('Data', fontsize=self.config.label_fontsize)
            ax2.grid(True, alpha=self.config.grid_alpha)
            
            plt.tight_layout()
        else:
            plt.figure(figsize=figsize)
            plt.plot(
                self.daily.index,
                self.daily['cum_profit'],
                color=colors['equity'],
                linewidth=2,
                marker='o',
                markersize=4
            )
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            plt.title(
                f'Curva de Equity Diária - {self.symbol} ({self.timeframe})',
                fontsize=self.config.title_fontsize
            )
            plt.xlabel('Data', fontsize=self.config.label_fontsize)
            plt.ylabel('Resultado Acumulado (R$)', fontsize=self.config.label_fontsize)
            plt.grid(True, alpha=self.config.grid_alpha)
            plt.tight_layout()
        
        return plt

    def plot_daily_returns(self, figsize: Optional[Tuple[int, int]] = None):
        """
        Plot daily returns as bar chart with cumulative line overlay.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure dimensions (width, height)
        
        Returns
        -------
        matplotlib.pyplot
        """
        figsize = figsize or (14, 8)
        colors = self.config.colors
        
        fig, ax1 = plt.subplots(figsize=figsize)
        
        # Daily profit bars
        bar_colors = [colors['buy'] if x >= 0 else colors['sell'] for x in self.daily['profit']]
        bars = ax1.bar(
            range(len(self.daily)),
            self.daily['profit'],
            color=bar_colors,
            alpha=0.7,
            label='Resultado Diário'
        )
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax1.set_ylabel('Resultado Diário (R$)', fontsize=self.config.label_fontsize)
        ax1.set_xlabel('Data', fontsize=self.config.label_fontsize)
        
        # Cumulative line on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(
            range(len(self.daily)),
            self.daily['cum_profit'],
            color=colors['total'],
            linewidth=2,
            label='Acumulado'
        )
        ax2.set_ylabel('Resultado Acumulado (R$)', fontsize=self.config.label_fontsize)
        
        # X-axis labels
        ax1.set_xticks(range(len(self.daily)))
        ax1.set_xticklabels(
            [d.strftime('%d/%m') for d in self.daily.index],
            rotation=45,
            ha='right',
            fontsize=8
        )
        
        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax1.set_title(
            f'Resultado Diário - {self.symbol} ({self.timeframe})',
            fontsize=self.config.title_fontsize
        )
        ax1.grid(True, axis='y', alpha=self.config.grid_alpha)
        
        plt.tight_layout()
        return plt

    def plot_daily_by_position(self, figsize: Optional[Tuple[int, int]] = None):
        """
        Plot daily equity curve separated by buy and sell positions.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure dimensions (width, height)
        
        Returns
        -------
        matplotlib.pyplot
        """
        figsize = figsize or self.config.figsize
        colors = self.config.colors
        
        plt.figure(figsize=figsize)
        
        plt.plot(
            self.daily.index,
            self.daily['cum_profit'],
            label='Total',
            color=colors['total'],
            linewidth=2,
            marker='o',
            markersize=4
        )
        plt.plot(
            self.daily.index,
            self.daily['cum_profit_buy'],
            label='Compras',
            color=colors['buy'],
            linewidth=1.5,
            marker='s',
            markersize=3
        )
        plt.plot(
            self.daily.index,
            self.daily['cum_profit_sell'],
            label='Vendas',
            color=colors['sell'],
            linewidth=1.5,
            marker='^',
            markersize=3
        )
        
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.title(
            f'Retorno Acumulado Diário por Tipo de Posição - {self.symbol} ({self.timeframe})',
            fontsize=self.config.title_fontsize
        )
        plt.xlabel('Data', fontsize=self.config.label_fontsize)
        plt.ylabel('Resultado Acumulado (R$)', fontsize=self.config.label_fontsize)
        plt.grid(True, alpha=self.config.grid_alpha)
        plt.legend()
        plt.tight_layout()
        
        return plt

    def plot_daily_metrics(self, figsize: Optional[Tuple[int, int]] = None):
        """
        Plot multiple daily metrics: trades count, win rate, and profit.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure dimensions (width, height)
        
        Returns
        -------
        matplotlib.pyplot
        """
        figsize = figsize or (14, 10)
        colors = self.config.colors
        
        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
        
        x = range(len(self.daily))
        
        # 1. Number of trades per day
        axes[0].bar(x, self.daily['num_buys'], label='Compras', color=colors['buy'], alpha=0.7)
        axes[0].bar(x, self.daily['num_sells'], bottom=self.daily['num_buys'],
                    label='Vendas', color=colors['sell'], alpha=0.7)
        axes[0].set_ylabel('Nº de Trades', fontsize=self.config.label_fontsize)
        axes[0].set_title('Número de Trades por Dia', fontsize=self.config.title_fontsize - 2)
        axes[0].legend(loc='upper right')
        axes[0].grid(True, axis='y', alpha=self.config.grid_alpha)
        
        # 2. Win rate per day with moving average
        axes[1].bar(x, self.daily['win_rate'], color=colors['equity'], alpha=0.5, label='Taxa de Acerto')
        axes[1].plot(x, self.daily['win_rate_ma7'], color=colors['total'],
                     linewidth=2, label='Média 7 dias')
        axes[1].axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        axes[1].set_ylabel('Taxa de Acerto (%)', fontsize=self.config.label_fontsize)
        axes[1].set_title('Taxa de Acerto Diária', fontsize=self.config.title_fontsize - 2)
        axes[1].legend(loc='upper right')
        axes[1].grid(True, axis='y', alpha=self.config.grid_alpha)
        axes[1].set_ylim(0, 100)
        
        # 3. Daily profit with moving averages
        bar_colors = [colors['buy'] if p >= 0 else colors['sell'] for p in self.daily['profit']]
        axes[2].bar(x, self.daily['profit'], color=bar_colors, alpha=0.6, label='Resultado')
        axes[2].plot(x, self.daily['profit_ma7'], color=colors['total'],
                     linewidth=2, label='Média 7 dias')
        axes[2].plot(x, self.daily['profit_ma30'], color=colors['neutral'],
                     linewidth=2, linestyle='--', label='Média 30 dias')
        axes[2].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        axes[2].set_ylabel('Resultado (R$)', fontsize=self.config.label_fontsize)
        axes[2].set_xlabel('Data', fontsize=self.config.label_fontsize)
        axes[2].set_title('Resultado Diário com Médias Móveis', fontsize=self.config.title_fontsize - 2)
        axes[2].legend(loc='upper right')
        axes[2].grid(True, axis='y', alpha=self.config.grid_alpha)
        
        # X-axis labels
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(
            [d.strftime('%d/%m') for d in self.daily.index],
            rotation=45,
            ha='right',
            fontsize=8
        )
        
        fig.suptitle(
            f'Métricas Diárias - {self.symbol} ({self.timeframe})',
            fontsize=self.config.title_fontsize + 2
        )
        plt.tight_layout()
        
        return plt

    def plot_daily_heatmap(self, figsize: Optional[Tuple[int, int]] = None):
        """
        Plot calendar heatmap of daily profits.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure dimensions (width, height)
        
        Returns
        -------
        matplotlib.pyplot
        """
        figsize = figsize or (14, 6)
        
        # Prepare data for heatmap
        daily = self.daily.copy()
        daily['weekday'] = daily.index.weekday
        daily['week'] = daily.index.isocalendar().week
        daily['year_week'] = daily.index.strftime('%Y-W%V')
        
        # Create pivot table
        pivot = daily.pivot_table(
            values='profit',
            index='weekday',
            columns='year_week',
            aggfunc='sum'
        )
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        # Custom colormap (red for negative, green for positive)
        from matplotlib.colors import TwoSlopeNorm
        
        vmax = max(abs(pivot.values.min()), abs(pivot.values.max()))
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        
        im = ax.imshow(pivot.values, cmap='RdYlGn', norm=norm, aspect='auto')
        
        # Labels
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom'][:len(pivot.index)])
        
        # X-axis: show only some labels if too many
        num_weeks = len(pivot.columns)
        if num_weeks > 20:
            step = num_weeks // 10
            ax.set_xticks(range(0, num_weeks, step))
            ax.set_xticklabels([pivot.columns[i] for i in range(0, num_weeks, step)], rotation=45, ha='right')
        else:
            ax.set_xticks(range(num_weeks))
            ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Resultado (R$)', fontsize=self.config.label_fontsize)
        
        ax.set_title(
            f'Heatmap de Resultados Diários - {self.symbol} ({self.timeframe})',
            fontsize=self.config.title_fontsize
        )
        ax.set_xlabel('Semana', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Dia da Semana', fontsize=self.config.label_fontsize)
        
        plt.tight_layout()
        return plt

    def plot_daily_dashboard(self, figsize: Optional[Tuple[int, int]] = None):
        """
        Create a comprehensive daily dashboard.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure dimensions (width, height)
        
        Returns
        -------
        matplotlib.pyplot
        """
        figsize = figsize or (16, 12)
        colors = self.config.colors
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
        
        x = range(len(self.daily))
        
        # 1. Daily equity curve (top, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(
            self.daily.index,
            self.daily['cum_profit'],
            color=colors['equity'],
            linewidth=2,
            marker='o',
            markersize=3
        )
        ax1.fill_between(
            self.daily.index,
            self.daily['cum_profit'],
            0,
            where=self.daily['cum_profit'] >= 0,
            color=colors['buy'],
            alpha=0.2
        )
        ax1.fill_between(
            self.daily.index,
            self.daily['cum_profit'],
            0,
            where=self.daily['cum_profit'] < 0,
            color=colors['sell'],
            alpha=0.2
        )
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title('Equity Diária', fontsize=12)
        ax1.set_ylabel('Resultado (R$)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Summary statistics (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        
        total_days = len(self.daily)
        winning_days = len(self.daily[self.daily['profit'] > 0])
        total_profit = self.daily['profit'].sum()
        avg_daily = self.daily['profit'].mean()
        max_dd = abs(self.daily['drawdown'].min())
        
        stats_text = f"""
        RESUMO DIÁRIO
        {'─' * 25}
        Dias operados: {total_days}
        Dias positivos: {winning_days} ({winning_days/total_days*100:.1f}%)
        
        Lucro total: R$ {total_profit:.2f}
        Média diária: R$ {avg_daily:.2f}
        
        Max Drawdown: R$ {max_dd:.2f}
        
        Melhor dia: R$ {self.daily['profit'].max():.2f}
        Pior dia: R$ {self.daily['profit'].min():.2f}
        """
        ax2.text(0.1, 0.5, stats_text, transform=ax2.transAxes, fontsize=10,
                 verticalalignment='center', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. Daily returns bar chart (middle, spans 2 columns)
        ax3 = fig.add_subplot(gs[1, :2])
        bar_colors = [colors['buy'] if p >= 0 else colors['sell'] for p in self.daily['profit']]
        ax3.bar(x, self.daily['profit'], color=bar_colors, alpha=0.7)
        ax3.plot(x, self.daily['profit_ma7'], color=colors['total'], linewidth=2, label='MA7')
        ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax3.set_title('Resultado Diário', fontsize=12)
        ax3.set_ylabel('Resultado (R$)')
        ax3.legend(loc='upper right')
        ax3.grid(True, axis='y', alpha=0.3)
        
        # 4. Win rate (middle right)
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.bar(x, self.daily['win_rate'], color=colors['equity'], alpha=0.6)
        ax4.plot(x, self.daily['win_rate_ma7'], color=colors['total'], linewidth=2)
        ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
        ax4.set_title('Taxa de Acerto (%)', fontsize=12)
        ax4.set_ylim(0, 100)
        ax4.grid(True, axis='y', alpha=0.3)
        
        # 5. By position (bottom left, spans 2 columns)
        ax5 = fig.add_subplot(gs[2, :2])
        ax5.plot(self.daily.index, self.daily['cum_profit'], label='Total',
                 color=colors['total'], linewidth=2)
        ax5.plot(self.daily.index, self.daily['cum_profit_buy'], label='Compras',
                 color=colors['buy'], linewidth=1.5)
        ax5.plot(self.daily.index, self.daily['cum_profit_sell'], label='Vendas',
                 color=colors['sell'], linewidth=1.5)
        ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax5.set_title('Resultado por Posição', fontsize=12)
        ax5.set_xlabel('Data')
        ax5.set_ylabel('Resultado (R$)')
        ax5.legend(loc='upper left')
        ax5.grid(True, alpha=0.3)
        
        # 6. Drawdown (bottom right)
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.fill_between(x, self.daily['drawdown'], 0, color=colors['drawdown'], alpha=0.3)
        ax6.plot(x, self.daily['drawdown'], color=colors['drawdown'], linewidth=1)
        ax6.set_title('Drawdown', fontsize=12)
        ax6.set_xlabel('Dia')
        ax6.set_ylabel('Drawdown (R$)')
        ax6.grid(True, alpha=0.3)
        
        fig.suptitle(
            f'Dashboard Diário - {self.symbol} ({self.timeframe})',
            fontsize=16,
            fontweight='bold'
        )
        
        plt.tight_layout()
        return plt

    def print_daily_summary(self):
        """Print formatted daily statistics summary."""
        daily = self.daily
        
        total_days = len(daily)
        winning_days = len(daily[daily['profit'] > 0])
        losing_days = len(daily[daily['profit'] < 0])
        
        print("=" * 60)
        print(f"RESUMO DIÁRIO - {self.symbol} ({self.timeframe})")
        print("=" * 60)
        
        print(f"\n{'PERÍODO':^60}")
        print("-" * 60)
        print(f"Data inicial:         {daily.index.min().strftime('%Y-%m-%d')}")
        print(f"Data final:           {daily.index.max().strftime('%Y-%m-%d')}")
        print(f"Total de dias:        {total_days}")
        
        print(f"\n{'RESULTADOS DIÁRIOS':^60}")
        print("-" * 60)
        print(f"Dias positivos:       {winning_days} ({winning_days/total_days*100:.1f}%)")
        print(f"Dias negativos:       {losing_days} ({losing_days/total_days*100:.1f}%)")
        print(f"Lucro total:          R$ {daily['profit'].sum():.2f}")
        print(f"Média diária:         R$ {daily['profit'].mean():.2f}")
        print(f"Desvio padrão:        R$ {daily['profit'].std():.2f}")
        print(f"Melhor dia:           R$ {daily['profit'].max():.2f}")
        print(f"Pior dia:             R$ {daily['profit'].min():.2f}")
        
        print(f"\n{'TRADES':^60}")
        print("-" * 60)
        print(f"Total de trades:      {daily['num_trades'].sum()}")
        print(f"Média trades/dia:     {daily['num_trades'].mean():.1f}")
        print(f"Total compras:        {daily['num_buys'].sum()}")
        print(f"Total vendas:         {daily['num_sells'].sum()}")
        
        print(f"\n{'RISCO':^60}")
        print("-" * 60)
        print(f"Max Drawdown:         R$ {abs(daily['drawdown'].min()):.2f}")
        print(f"Max Drawdown (%):     {abs(daily['drawdown_pct'].min())*100:.2f}%")
        
        # Sharpe ratio (simplified, assuming risk-free = 0)
        if daily['profit'].std() > 0:
            sharpe = (daily['profit'].mean() / daily['profit'].std()) * np.sqrt(252)
            print(f"Sharpe Ratio (anual): {sharpe:.2f}")
        
        print("=" * 60)

    # =========================================================================
    # SUMMARY STATISTICS
    # =========================================================================
    def get_statistics(self) -> dict:
        """
        Calculate and return comprehensive backtest statistics.
        
        Returns
        -------
        dict
            Dictionary with all relevant statistics
        """
        trades = self.trades
        results = self.results
        
        total_trades = len(trades)
        if total_trades == 0:
            return {"error": "No trades found"}
        
        wins = len(trades[trades['profit'] > 0])
        losses = len(trades[trades['profit'] < 0])
        
        stats = {
            # Trade counts
            'total_trades': total_trades,
            'winning_trades': wins,
            'losing_trades': losses,
            'win_rate': wins / total_trades * 100,
            
            # Trade outcomes
            'take_profits': len(trades[trades['trade_status'] == 1]),
            'stop_losses': len(trades[trades['trade_status'] == -1]),
            'end_of_day': len(trades[trades['trade_status'] == 0]),
            
            # Profit metrics
            'total_profit': trades['profit'].sum(),
            'total_points': trades['trade_points'].sum(),
            'avg_profit': trades['profit'].mean(),
            'avg_win': trades.loc[trades['profit'] > 0, 'profit'].mean() if wins > 0 else 0,
            'avg_loss': trades.loc[trades['profit'] < 0, 'profit'].mean() if losses > 0 else 0,
            'max_profit': trades['profit'].max(),
            'max_loss': trades['profit'].min(),
            
            # Risk metrics
            'max_drawdown': abs(results['drawdown'].min()),
            'max_drawdown_pct': abs(results['drawdown_pct'].min()) * 100,
            'peak_profit': results['equity'].max(),
            
            # Ratios
            'profit_factor': None,
            'risk_reward': None
        }
        
        # Calculate profit factor
        gross_profit = trades.loc[trades['profit'] > 0, 'profit'].sum()
        gross_loss = abs(trades.loc[trades['profit'] < 0, 'profit'].sum())
        if gross_loss > 0:
            stats['profit_factor'] = gross_profit / gross_loss
        
        # Calculate risk/reward
        if stats['avg_loss'] != 0:
            stats['risk_reward'] = abs(stats['avg_win'] / stats['avg_loss'])
        
        return stats

    def print_statistics(self):
        """Print formatted statistics summary."""
        stats = self.get_statistics()
        
        if 'error' in stats:
            print(stats['error'])
            return
        
        print("=" * 60)
        print(f"RESUMO DO BACKTEST - {self.symbol} ({self.timeframe})")
        print("=" * 60)
        
        print(f"\n{'TRADES':^60}")
        print("-" * 60)
        print(f"Total de trades:      {stats['total_trades']}")
        print(f"  Take Profits (TP):  {stats['take_profits']} ({stats['take_profits']/stats['total_trades']*100:.1f}%)")
        print(f"  Stop Losses (SL):   {stats['stop_losses']} ({stats['stop_losses']/stats['total_trades']*100:.1f}%)")
        print(f"  Fim do Dia (EOD):   {stats['end_of_day']} ({stats['end_of_day']/stats['total_trades']*100:.1f}%)")
        
        print(f"\n{'RESULTADOS':^60}")
        print("-" * 60)
        print(f"Taxa de acerto:       {stats['win_rate']:.1f}%")
        print(f"Lucro total:          R$ {stats['total_profit']:.2f}")
        print(f"Pontos totais:        {stats['total_points']:.0f}")
        print(f"Lucro médio:          R$ {stats['avg_profit']:.2f}")
        print(f"Média ganhos:         R$ {stats['avg_win']:.2f}")
        print(f"Média perdas:         R$ {stats['avg_loss']:.2f}")
        
        print(f"\n{'RISCO':^60}")
        print("-" * 60)
        print(f"Máximo drawdown:      R$ {stats['max_drawdown']:.2f}")
        print(f"Máximo drawdown (%):  {stats['max_drawdown_pct']:.2f}%")
        print(f"Pico de lucro:        R$ {stats['peak_profit']:.2f}")
        print(f"Maior ganho:          R$ {stats['max_profit']:.2f}")
        print(f"Maior perda:          R$ {stats['max_loss']:.2f}")
        
        print(f"\n{'RATIOS':^60}")
        print("-" * 60)
        if stats['profit_factor']:
            print(f"Profit Factor:        {stats['profit_factor']:.2f}")
        else:
            print("Profit Factor:        N/A")
        if stats['risk_reward']:
            print(f"Risk/Reward:          {stats['risk_reward']:.2f}")
        else:
            print("Risk/Reward:          N/A")
        
        print("=" * 60)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================
def quick_plot(
    results: pd.DataFrame, 
    trades: pd.DataFrame,
    symbol: str = 'WIN',
    timeframe: str = 'Tick',
    daily: bool = False
):
    """
    Quick visualization of backtest results.
    
    Parameters
    ----------
    results : pd.DataFrame
        Full results DataFrame from run_backtest()
    trades : pd.DataFrame
        Trades-only DataFrame from run_backtest()
    symbol : str
        Trading symbol name
    timeframe : str
        Timeframe description
    daily : bool
        If True, show daily dashboard instead of tick-level (default: False)
    
    Example
    -------
    >>> results, trades = run_backtest(tick_data, config)
    >>> quick_plot(results, trades, daily=True)
    >>> plt.show()
    """
    viz = BacktestVisualizer(results, trades, symbol, timeframe)
    
    if daily:
        viz.print_daily_summary()
        viz.plot_daily_dashboard()
    else:
        viz.print_statistics()
        viz.plot_dashboard()
    
    return viz


def quick_daily_plot(
    results: pd.DataFrame, 
    trades: pd.DataFrame,
    symbol: str = 'WIN',
    timeframe: str = 'Tick'
):
    """
    Quick daily visualization of backtest results.
    
    Parameters
    ----------
    results : pd.DataFrame
        Full results DataFrame from run_backtest()
    trades : pd.DataFrame
        Trades-only DataFrame from run_backtest()
    symbol : str
        Trading symbol name
    timeframe : str
        Timeframe description
    
    Example
    -------
    >>> results, trades = run_backtest(tick_data, config)
    >>> quick_daily_plot(results, trades)
    >>> plt.show()
    """
    return quick_plot(results, trades, symbol, timeframe, daily=True)


# =============================================================================
# MAIN EXECUTION (EXAMPLE)
# =============================================================================
if __name__ == "__main__":
    # This example shows how to use the visualizer
    # You need to import and run the backtest first
    
    print("=" * 60)
    print("VISUALIZATION MODULE FOR DYNAMIC GRID TRADING")
    print("=" * 60)
    print("""
    Usage Example:
    --------------
    from backtest_dynamic_grid import run_backtest, load_date_range, StrategyConfig
    from visualization_dynamic_grid import BacktestVisualizer, quick_plot, quick_daily_plot
    
    # Load data and run backtest
    tick_data = load_date_range(base_path, '2024-01-01', '2024-12-31')
    config = StrategyConfig(distance=70, stop_loss=50, take_profit=100)
    results, trades = run_backtest(tick_data, config)
    
    # Option 1: Quick visualization (tick-level)
    viz = quick_plot(results, trades, symbol='WIN', timeframe='Tick')
    plt.show()
    
    # Option 2: Quick daily visualization
    viz = quick_daily_plot(results, trades, symbol='WIN', timeframe='Tick')
    plt.show()
    
    # Option 3: Individual plots
    viz = BacktestVisualizer(results, trades, symbol='WIN', timeframe='Tick')
    
    # Tick-level plots
    viz.plot_equity_curve()
    viz.plot_by_position()
    viz.plot_profit_by_hour()
    viz.plot_dashboard()
    
    # Daily plots (NEW)
    viz.plot_daily_equity()
    viz.plot_daily_returns()
    viz.plot_daily_by_position()
    viz.plot_daily_metrics()
    viz.plot_daily_heatmap()
    viz.plot_daily_dashboard()
    
    # Get statistics
    viz.print_statistics()       # Tick-level stats
    viz.print_daily_summary()    # Daily stats
    
    # Access daily data directly
    daily_df = viz.get_daily_data()
    print(daily_df.head())
    
    plt.show()
    """)