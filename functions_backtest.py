def load_parquet_data(
    path: str,
    timestamp_col: str = 'data_trade',
    price_col: str = 'Preço'
) -> pd.DataFrame:
    """
    Carrega dados tick a tick de arquivo(s) Parquet.
    
    Parâmetros:
    -----------
    path : str
        Caminho para o arquivo ou diretório Parquet
    timestamp_col : str
        Nome da coluna de timestamp (default: 'data_trade')
    price_col : str
        Nome da coluna de preço (default: 'Preço')
    
    Retorna:
    --------
    pd.DataFrame com colunas 'timestamp' e 'price'
    
    Exemplo de uso:
    ---------------
    # Carregar um dia específico
    data = '20240202'
    path = f'C:/Users/.../WIN.parquet/anomesdia={data}'
    tick_data = load_parquet_data(path)
    
    # Carregar múltiplos dias
    paths = [f'C:/Users/.../WIN.parquet/anomesdia={d}' for d in ['20240201', '20240202']]
    dfs = [load_parquet_data(p) for p in paths]
    tick_data = pd.concat(dfs, ignore_index=True)
    """
    df = pd.read_parquet(path)
    
    # Validar colunas
    if timestamp_col not in df.columns:
        raise ValueError(f"Coluna '{timestamp_col}' não encontrada. Colunas disponíveis: {list(df.columns)}")
    if price_col not in df.columns:
        raise ValueError(f"Coluna '{price_col}' não encontrada. Colunas disponíveis: {list(df.columns)}")
    
    # Criar DataFrame padronizado
    result = pd.DataFrame({
        'timestamp': pd.to_datetime(df[timestamp_col]),
        'price': df[price_col].astype(float)
    })
    
    # Ordenar por timestamp
    result = result.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Dados carregados:")
    print(f"  - Período: {result['timestamp'].min()} até {result['timestamp'].max()}")
    print(f"  - Total de ticks: {len(result):,}")
    print(f"  - Preço mín: {result['price'].min():,.0f}")
    print(f"  - Preço máx: {result['price'].max():,.0f}")
    
    return result


def load_multiple_days(
    base_path: str,
    dates: List[str],
    timestamp_col: str = 'data_trade',
    price_col: str = 'Preço'
) -> pd.DataFrame:
    """
    Carrega dados de múltiplos dias de uma estrutura particionada.
    
    Parâmetros:
    -----------
    base_path : str
        Caminho base do Parquet (ex: 'C:/Users/.../WIN.parquet')
    dates : List[str]
        Lista de datas no formato 'YYYYMMDD'
    timestamp_col : str
        Nome da coluna de timestamp
    price_col : str
        Nome da coluna de preço
    
    Retorna:
    --------
    pd.DataFrame concatenado com todos os dias
    
    Exemplo:
    --------
    base_path = 'C:/Users/User/OneDrive/Documentos/rnt/Finance/Trading Projects/00.database/tick data/WIN.parquet'
    dates = ['20240201', '20240202', '20240205']
    tick_data = load_multiple_days(base_path, dates)
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
            print(f"✗ {date}: Erro ao carregar - {e}")
    
    if not all_dfs:
        raise ValueError("Nenhum dado foi carregado")
    
    result = pd.concat(all_dfs, ignore_index=True)
    result = result.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\nTotal consolidado: {len(result):,} ticks")
    print(f"Período: {result['timestamp'].min()} até {result['timestamp'].max()}")
    
    return result


def signal(df, distance):
    
    '''
    Function to compute the signal of the trading, i.e. when the pending order with be activated
    '''

    # Distance of the price for pending orders
    distance = 70

    # Flag for pending orders
    df['flag_pending_order'] = 0
    df.loc[(df['timestamp'].dt.second == 0) & (df['timestamp'].dt.second != 0).shift(1), 'flag_pending_order'] = 1

    # Reference Price
    df.loc[df['flag_pending_order'] == 1, 'price_ref'] = df.loc[df['flag_pending_order'] == 1, 'price']
    df['price_ref'] = df['price_ref'].ffill()
    df['delta_price'] = df['price'] - df['price_ref']

    # Create order period groups (increments each time a new pending order is placed)
    df['order_period'] = df['flag_pending_order'].cumsum()

    # Detect threshold crossings
    df['crossing_buy'] = (df['delta_price'] >= distance) & (df['delta_price'].shift(1) < distance)
    df['crossing_sell'] = (df['delta_price'] <= -distance) & (df['delta_price'].shift(1) > -distance)

    # Flag buy: only the FIRST crossing within each order period
    df['flag_buy'] = 0
    first_buy_crossings = df[df['crossing_buy']].groupby('order_period').head(1).index
    df.loc[first_buy_crossings, 'flag_buy'] = 1

    # Flag sell: only the FIRST crossing within each order period
    df['flag_sell'] = 0
    first_sell_crossings = df[df['crossing_sell']].groupby('order_period').head(1).index
    df.loc[first_sell_crossings, 'flag_sell'] = 1

    # Clean up helper columns
    df.drop(columns=['order_period', 'crossing_buy', 'crossing_sell'], inplace=True)
    
    return df


import numpy as np
from numba import jit

@jit(nopython=True)
def simulate_trades(prices, flag_buy, flag_sell, day_idx_final, sl, tp):
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
                
                # End of Day - compare j to row index, not day index
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


# Prepare arrays with explicit dtypes
prices = df['price'].values.astype(np.float64)
flag_buy = df['flag_buy'].values.astype(np.int64)
flag_sell = df['flag_sell'].values.astype(np.int64)

# FIXED: day_idx_final should be the ROW INDEX of the last bar of each day
df['row_idx'] = np.arange(len(df), dtype=np.int64)
df['day_idx_final'] = df.groupby(df['timestamp'].dt.date)['row_idx'].transform('last')

day_idx_final = df['day_idx_final'].values.astype(np.int64)

# Parameters
sl = 50
tp = 100
point_value = 0.20

# Run simulation (removed day_idx parameter)
trade_status, trade_entry_price, trade_exit_price, trade_points, exit_idx = simulate_trades(
    prices, flag_buy, flag_sell, day_idx_final, sl, tp
)

# Assign results
df['trade_status'] = trade_status
df['trade_entry_price'] = trade_entry_price
df['trade_exit_price'] = trade_exit_price
df['trade_points'] = trade_points
df['exit_idx'] = exit_idx
df['profit'] = trade_points * point_value
df['accumulated_profit'] = df['profit'].cumsum()

# Filter only trades (entries with non-zero status)
trades = df[(df['flag_buy'] == 1) | (df['flag_sell'] == 1)].copy()
trades = trades[trades['trade_status'] != 0]


# Summary statistics
print("=" * 50)
print("TRADE SUMMARY")
print("=" * 50)

total_trades = len(trades)
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
print(f"Profit factor:       {abs(avg_win * wins / (avg_loss * losses)):.2f}" if losses > 0 and avg_loss != 0 else "Profit factor: N/A")
print("-" * 50)
print(f"Total profit:        R$ {total_profit:.2f}")
print(f"Max single profit:   R$ {max_profit:.2f}")
print(f"Max single loss:     R$ {max_loss:.2f}")
print(f"Peak profit:         R$ {peak_profit:.2f}")
print(f"Max drawdown:        R$ {max_drawdown:.2f}")
print("=" * 50)

# Daily breakdown
daily_stats = trades.groupby(df.loc[trades.index, 'timestamp'].dt.date).agg({
    'trade_points': ['count', 'sum'],
    'profit': 'sum'
}).round(2)
daily_stats.columns = ['trades', 'points', 'profit_R$']
daily_stats['accumulated'] = daily_stats['profit_R$'].cumsum()

print("\nDAILY BREAKDOWN")
print(daily_stats.to_string())