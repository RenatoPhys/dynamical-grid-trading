"""
Day Trading Backtest Strategy
==============================
Strategy: Every 1 minute, set two pending orders (buy/sell) at distance X from the last price.
When price crosses the threshold, the order is triggered.

Author: Renato Critelli
"""

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import pandas as pd
from numba import jit
from typing import List, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class StrategyConfig:
    """Configuration parameters for the trading strategy."""
    distance: int = 70          # Distance for pending orders (points)
    stop_loss: int = 50         # Stop loss (points)
    take_profit: int = 100      # Take profit (points)
    point_value: float = 0.20   # Value per point in R$
    tc: float = 0.20*5          # trading costs
    invert_signals: bool = False  # If True: price up → sell, price down → buy (mean-reversion)
                                  # If False: price up → buy, price down → sell (momentum)
    
    # Data column names
    timestamp_col: str = 'data_trade'
    price_col: str = 'Preço'


# =============================================================================
# DATA LOADING
# =============================================================================
def load_parquet_data(
    path: str,
    timestamp_col: str = 'data_trade',
    price_col: str = 'Preço'
) -> pd.DataFrame:
    """
    Load tick-by-tick data from Parquet file(s).
    
    Parameters
    ----------
    path : str
        Path to Parquet file or directory
    timestamp_col : str
        Name of timestamp column
    price_col : str
        Name of price column
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'timestamp' and 'price' columns
    
    Example
    -------
    >>> data = '20240202'
    >>> path = f'C:/Users/.../WIN.parquet/anomesdia={data}'
    >>> tick_data = load_parquet_data(path)
    """
    df = pd.read_parquet(path)
    
    # Validate columns
    if timestamp_col not in df.columns:
        raise ValueError(f"Column '{timestamp_col}' not found. Available: {list(df.columns)}")
    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found. Available: {list(df.columns)}")
    
    # Create standardized DataFrame
    result = pd.DataFrame({
        'timestamp': pd.to_datetime(df[timestamp_col]),
        'price': df[price_col].astype(float)
    })
    
    # Sort by timestamp
    result = result.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Data loaded:")
    print(f"  - Period: {result['timestamp'].min()} to {result['timestamp'].max()}")
    print(f"  - Total ticks: {len(result):,}")
    print(f"  - Price range: {result['price'].min():,.0f} - {result['price'].max():,.0f}")
    
    return result


def load_multiple_days(
    base_path: str,
    dates: List[str],
    timestamp_col: str = 'data_trade',
    price_col: str = 'Preço'
) -> pd.DataFrame:
    """
    Load data from multiple days of a partitioned structure.
    
    Parameters
    ----------
    base_path : str
        Base Parquet path (e.g., 'C:/Users/.../WIN.parquet')
    dates : List[str]
        List of dates in 'YYYYMMDD' format
    timestamp_col : str
        Name of timestamp column
    price_col : str
        Name of price column
    
    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with all days
    
    Example
    -------
    >>> base_path = 'C:/Users/.../WIN.parquet'
    >>> dates = ['20240201', '20240202', '20240205']
    >>> tick_data = load_multiple_days(base_path, dates)
    """
    all_dfs = []
    
    for date in dates:
        path = f"{base_path}/anomesdia={date}"
        try:
            df = pd.read_parquet(path)
            df_clean = pd.DataFrame({
                'timestamp': pd.to_datetime(df[timestamp_col]),
                'price': df[price_col].astype(float)
            })
            all_dfs.append(df_clean)
            print(f"✓ {date}: {len(df_clean):,} ticks")
        except Exception as e:
            print(f"✗ {date}: Error loading - {e}")
    
    if not all_dfs:
        raise ValueError("No data was loaded")
    
    result = pd.concat(all_dfs, ignore_index=True)
    result = result.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\nTotal consolidated: {len(result):,} ticks")
    print(f"Period: {result['timestamp'].min()} to {result['timestamp'].max()}")
    
    return result


def load_date_range(
    base_path: str,
    start_date: str,
    end_date: str,
    timestamp_col: str = 'data_trade',
    price_col: str = 'Preço',
    skip_weekends: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load data from a date range.
    
    Parameters
    ----------
    base_path : str
        Base Parquet path (e.g., 'C:/Users/.../WIN.parquet')
    start_date : str
        Start date in 'YYYY-MM-DD' or 'YYYYMMDD' format
    end_date : str
        End date in 'YYYY-MM-DD' or 'YYYYMMDD' format
    timestamp_col : str
        Name of timestamp column
    price_col : str
        Name of price column
    skip_weekends : bool
        If True, skip Saturdays and Sundays (default: True)
    verbose : bool
        If True, print loading progress (default: True)
    
    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with all available days
    
    Example
    -------
    >>> base_path = 'C:/Users/.../WIN.parquet'
    >>> tick_data = load_date_range(base_path, '2024-01-01', '2024-12-31')
    """
    # Parse dates
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Generate all dates in range
    all_dates = pd.date_range(start=start, end=end, freq='D')
    
    if skip_weekends:
        # Filter to business days (Mon-Fri)
        all_dates = all_dates[all_dates.weekday < 5]
    
    all_dfs = []
    loaded_count = 0
    skipped_count = 0
    
    for date in all_dates:
        date_str = date.strftime('%Y%m%d')
        path = f"{base_path}/anomesdia={date_str}"
        
        try:
            df = pd.read_parquet(path)
            df_clean = pd.DataFrame({
                'timestamp': pd.to_datetime(df[timestamp_col]),
                'price': df[price_col].astype(float)
            })
            all_dfs.append(df_clean)
            loaded_count += 1
            if verbose:
                print(f"✓ {date_str}: {len(df_clean):,} ticks")
        except Exception:
            skipped_count += 1
            # Silently skip missing dates (holidays, etc.)
            pass
    
    if not all_dfs:
        raise ValueError(f"No data found between {start_date} and {end_date}")
    
    result = pd.concat(all_dfs, ignore_index=True)
    result = result.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\n{'='*50}")
    print(f"LOADING SUMMARY")
    print(f"{'='*50}")
    print(f"Date range:      {start_date} to {end_date}")
    print(f"Days loaded:     {loaded_count}")
    print(f"Days skipped:    {skipped_count} (holidays/missing)")
    print(f"Total ticks:     {len(result):,}")
    print(f"Period:          {result['timestamp'].min()} to {result['timestamp'].max()}")
    print(f"{'='*50}")
    
    return result


# =============================================================================
# SIGNAL GENERATION
# =============================================================================
def generate_signals(df: pd.DataFrame, distance: int, invert_signals: bool = False) -> pd.DataFrame:
    """
    Generate trading signals based on pending order strategy.
    
    Strategy: Every minute (when second changes to 0), place pending orders
    at +/- distance from current price. Signal when price crosses threshold.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'timestamp' and 'price' columns
    distance : int
        Distance in points for pending orders
    invert_signals : bool
        If False (default): price up → buy, price down → sell (momentum/breakout)
        If True: price up → sell, price down → buy (mean-reversion)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with signal columns added
    """
    df = df.copy()
    
    # Flag for new pending orders (every minute change)
    df['flag_pending_order'] = 0
    mask = (df['timestamp'].dt.second == 0) & (df['timestamp'].dt.second != 0).shift(1, fill_value=False)
    df.loc[mask, 'flag_pending_order'] = 1
    
    # Reference price (price at pending order creation)
    df['price_ref'] = np.nan
    df.loc[df['flag_pending_order'] == 1, 'price_ref'] = df.loc[df['flag_pending_order'] == 1, 'price']
    df['price_ref'] = df['price_ref'].ffill()
    
    # Delta from reference
    df['delta_price'] = df['price'] - df['price_ref']
    
    # Create order period groups
    df['order_period'] = df['flag_pending_order'].cumsum()
    
    # Detect threshold crossings
    crossing_up = (df['delta_price'] >= distance) & (df['delta_price'].shift(1) < distance)
    crossing_down = (df['delta_price'] <= -distance) & (df['delta_price'].shift(1) > -distance)
    
    # Assign buy/sell based on invert_signals flag
    if invert_signals:
        # Mean-reversion: price up → sell, price down → buy
        df['crossing_buy'] = crossing_down
        df['crossing_sell'] = crossing_up
    else:
        # Momentum/breakout: price up → buy, price down → sell
        df['crossing_buy'] = crossing_up
        df['crossing_sell'] = crossing_down
    
    # Flag buy: only FIRST crossing within each order period
    df['flag_buy'] = 0
    first_buy = df[df['crossing_buy']].groupby('order_period').head(1).index
    df.loc[first_buy, 'flag_buy'] = 1
    
    # Flag sell: only FIRST crossing within each order period
    df['flag_sell'] = 0
    first_sell = df[df['crossing_sell']].groupby('order_period').head(1).index
    df.loc[first_sell, 'flag_sell'] = 1
    
    # Clean up helper columns
    df.drop(columns=['order_period', 'crossing_buy', 'crossing_sell'], inplace=True)
    
    return df


# =============================================================================
# TRADE SIMULATION (Numba-optimized)
# =============================================================================
@jit(nopython=True)
def _simulate_trades_numba(
    prices: np.ndarray,
    flag_buy: np.ndarray,
    flag_sell: np.ndarray,
    day_idx_final: np.ndarray,
    sl: float,
    tp: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba-optimized trade simulation.
    
    Parameters
    ----------
    prices : np.ndarray
        Array of prices
    flag_buy : np.ndarray
        Array of buy signals (1 = buy, 0 = no signal)
    flag_sell : np.ndarray
        Array of sell signals (1 = sell, 0 = no signal)
    day_idx_final : np.ndarray
        Array with the last row index for each day
    sl : float
        Stop loss in points
    tp : float
        Take profit in points
    
    Returns
    -------
    Tuple of arrays: (trade_status, entry_price, exit_price, points, exit_idx)
        trade_status: 1 = TP, -1 = SL, 0 = EOD
    """
    n = prices.shape[0]
    
    trade_status = np.zeros(n, dtype=np.float64)
    trade_entry_price = np.zeros(n, dtype=np.float64)
    trade_exit_price = np.zeros(n, dtype=np.float64)
    trade_points = np.zeros(n, dtype=np.float64)
    exit_idx = np.zeros(n, dtype=np.int64)
    
    for i in range(n):
        if flag_buy[i] == 1:
            entry_price = prices[i]
            for j in range(i + 1, n):
                pnl = prices[j] - entry_price
                
                # Stop Loss
                if pnl <= -sl:
                    trade_status[i] = -1
                    trade_entry_price[i] = entry_price
                    trade_exit_price[i] = entry_price - sl
                    trade_points[i] = -sl
                    exit_idx[i] = j
                    break
                
                # Take Profit
                if pnl >= tp:
                    trade_status[i] = 1
                    trade_entry_price[i] = entry_price
                    trade_exit_price[i] = entry_price + tp
                    trade_points[i] = tp
                    exit_idx[i] = j
                    break
                
                # End of Day
                if j >= day_idx_final[i]:
                    trade_status[i] = 0
                    trade_entry_price[i] = entry_price
                    trade_exit_price[i] = prices[j]
                    trade_points[i] = pnl
                    exit_idx[i] = j
                    break
                    
        elif flag_sell[i] == 1:
            entry_price = prices[i]
            for j in range(i + 1, n):
                pnl = entry_price - prices[j]
                
                # Stop Loss
                if pnl <= -sl:
                    trade_status[i] = -1
                    trade_entry_price[i] = entry_price
                    trade_exit_price[i] = entry_price + sl
                    trade_points[i] = -sl
                    exit_idx[i] = j
                    break
                
                # Take Profit
                if pnl >= tp:
                    trade_status[i] = 1
                    trade_entry_price[i] = entry_price
                    trade_exit_price[i] = entry_price - tp
                    trade_points[i] = tp
                    exit_idx[i] = j
                    break
                
                # End of Day
                if j >= day_idx_final[i]:
                    trade_status[i] = 0
                    trade_entry_price[i] = entry_price
                    trade_exit_price[i] = prices[j]
                    trade_points[i] = pnl
                    exit_idx[i] = j
                    break
    
    return trade_status, trade_entry_price, trade_exit_price, trade_points, exit_idx


def simulate_trades(df: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    """
    Run trade simulation on signal data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with signals (flag_buy, flag_sell)
    config : StrategyConfig
        Strategy configuration
    
    Returns
    -------
    pd.DataFrame
        DataFrame with trade results added
    """
    df = df.copy()
    
    # Prepare arrays
    prices = df['price'].values.astype(np.float64)
    flag_buy = df['flag_buy'].values.astype(np.int64)
    flag_sell = df['flag_sell'].values.astype(np.int64)
    
    # Calculate last row index for each day
    df['row_idx'] = np.arange(len(df), dtype=np.int64)
    df['day_idx_final'] = df.groupby(df['timestamp'].dt.date)['row_idx'].transform('last')
    day_idx_final = df['day_idx_final'].values.astype(np.int64)
    
    # Run simulation
    trade_status, entry_price, exit_price, trade_points, exit_idx = _simulate_trades_numba(
        prices, flag_buy, flag_sell, day_idx_final,
        config.stop_loss, config.take_profit
    )
    
    # Assign results
    df['trade_status'] = trade_status
    df['trade_entry_price'] = entry_price
    df['trade_exit_price'] = exit_price
    df['trade_points'] = trade_points
    df['exit_idx'] = exit_idx
    df['profit'] = trade_points * config.point_value - config.tc
    df['accumulated_profit'] = df['profit'].cumsum()
    
    # Clean up helper columns
    df.drop(columns=['row_idx', 'day_idx_final'], inplace=True)
    
    return df


# =============================================================================
# REPORTING
# =============================================================================
def get_trades(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only trade entries from simulation results."""
    trades = df[(df['flag_buy'] == 1) | (df['flag_sell'] == 1)].copy()
    trades = trades[trades['trade_entry_price'] != 0]  # Filter valid trades
    return trades


def print_summary(df: pd.DataFrame, trades: pd.DataFrame) -> None:
    """Print trade summary statistics."""
    print("=" * 50)
    print("TRADE SUMMARY")
    print("=" * 50)
    
    total_trades = len(trades)
    if total_trades == 0:
        print("No trades executed.")
        return
    
    tp_trades = len(trades[trades['trade_status'] == 1])
    sl_trades = len(trades[trades['trade_status'] == -1])
    eod_trades = len(trades[trades['trade_status'] == 0])
    
    wins = len(trades[trades['trade_points'] > 0])
    losses = len(trades[trades['trade_points'] < 0])
    breakeven = len(trades[trades['trade_points'] == 0])
    
    total_points = trades['trade_points'].sum()
    total_profit = trades['profit'].sum()
    
    avg_win = trades.loc[trades['trade_points'] > 0, 'trade_points'].mean() if wins > 0 else 0
    avg_loss = trades.loc[trades['trade_points'] < 0, 'trade_points'].mean() if losses > 0 else 0
    
    max_profit = trades['profit'].max()
    max_loss = trades['profit'].min()
    
    # Accumulated profit stats
    accumulated = df['accumulated_profit']
    max_drawdown = (accumulated.cummax() - accumulated).max()
    peak_profit = accumulated.max()
    
    print(f"Total trades:        {total_trades}")
    print(f"  Take Profits (TP): {tp_trades} ({tp_trades/total_trades*100:.1f}%)")
    print(f"  Stop Losses (SL):  {sl_trades} ({sl_trades/total_trades*100:.1f}%)")
    print(f"  End of Day (EOD):  {eod_trades} ({eod_trades/total_trades*100:.1f}%)")
    print("-" * 50)
    print(f"Wins:                {wins} ({wins/total_trades*100:.1f}%)")
    print(f"Losses:              {losses} ({losses/total_trades*100:.1f}%)")
    print(f"Breakeven:           {breakeven}")
    print("-" * 50)
    print(f"Total points:        {total_points:.0f}")
    print(f"Avg win (pts):       {avg_win:.1f}")
    print(f"Avg loss (pts):      {avg_loss:.1f}")
    
    if losses > 0 and avg_loss != 0:
        profit_factor = abs(avg_win * wins / (avg_loss * losses))
        print(f"Profit factor:       {profit_factor:.2f}")
    else:
        print("Profit factor:       N/A")
    
    print("-" * 50)
    print(f"Total profit:        R$ {total_profit:.2f}")
    print(f"Max single profit:   R$ {max_profit:.2f}")
    print(f"Max single loss:     R$ {max_loss:.2f}")
    print(f"Peak profit:         R$ {peak_profit:.2f}")
    print(f"Max drawdown:        R$ {max_drawdown:.2f}")
    print("=" * 50)


def print_daily_breakdown(df: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    """Print and return daily breakdown statistics."""
    daily_stats = trades.groupby(df.loc[trades.index, 'timestamp'].dt.date).agg({
        'trade_points': ['count', 'sum'],
        'profit': 'sum'
    }).round(2)
    daily_stats.columns = ['trades', 'points', 'profit_R$']
    daily_stats['accumulated'] = daily_stats['profit_R$'].cumsum()
    
    print("\nDAILY BREAKDOWN")
    print(daily_stats.to_string())
    
    return daily_stats


# =============================================================================
# MAIN BACKTEST FUNCTION
# =============================================================================
def run_backtest(
    tick_data: pd.DataFrame,
    config: Optional[StrategyConfig] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run complete backtest pipeline.
    
    Parameters
    ----------
    tick_data : pd.DataFrame
        Tick data with 'timestamp' and 'price' columns
    config : StrategyConfig, optional
        Strategy configuration (uses defaults if not provided)
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (full_results, trades_only)
    """
    if config is None:
        config = StrategyConfig()
    
    mode = "MEAN-REVERSION (inverted)" if config.invert_signals else "MOMENTUM (normal)"
    
    print(f"\nRunning backtest with:")
    print(f"  - Mode: {mode}")
    print(f"  - Distance: {config.distance} pts")
    print(f"  - Stop Loss: {config.stop_loss} pts")
    print(f"  - Take Profit: {config.take_profit} pts")
    print(f"  - Point Value: R$ {config.point_value}")
    print()
    
    # Generate signals
    df = generate_signals(tick_data, config.distance, config.invert_signals)
    
    # Simulate trades
    df = simulate_trades(df, config)
    
    # Extract trades
    trades = get_trades(df)
    
    # Print reports
    print_summary(df, trades)
    print_daily_breakdown(df, trades)
    
    return df, trades


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # =========================
    # CONFIGURATION
    # =========================
    config = StrategyConfig(
        distance=60,
        stop_loss=60,
        take_profit=60,
        point_value=0.20,
        tc = 0.2*5,
        invert_signals=True  # False = momentum (breakout), True = mean-reversion
    )
    
    # =========================
    # LOAD DATA
    # =========================
    base_path = 'C:/Users/User/OneDrive/Documentos/rnt/Finance/Trading Projects/00.database/tick data/WIN.parquet'
    
    # Option 1: Date range (recommended)
    tick_data = load_date_range(base_path, start_date='2024-02-02', end_date='2024-02-02')    
   
    # =========================
    # RUN BACKTEST
    # =========================
    results, trades = run_backtest(tick_data, config)
    
    # =========================
    # COMPARE BOTH MODES
    # =========================
    # Uncomment to compare momentum vs mean-reversion:
    #
    # print("\n" + "="*60)
    # print("COMPARING BOTH MODES")
    # print("="*60)
    #
    # config_momentum = StrategyConfig(distance=70, stop_loss=50, take_profit=100, invert_signals=False)
    # config_reversion = StrategyConfig(distance=70, stop_loss=50, take_profit=100, invert_signals=True)
    #
    # results_mom, trades_mom = run_backtest(tick_data, config_momentum)
    # results_rev, trades_rev = run_backtest(tick_data, config_reversion)
    
    # =========================
    # OPTIONAL: SAVE RESULTS
    # =========================
    # trades.to_csv('backtest_trades.csv', index=False)
    # results.to_parquet('backtest_full_results.parquet')