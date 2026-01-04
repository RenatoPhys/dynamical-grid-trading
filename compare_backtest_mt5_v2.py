"""
Backtest vs MetaTrader 5 Trade Comparison - Enhanced Version
=============================================================
Script para comparar resultados do backtest Python com trades executados no MT5.
Inclui conexão direta ao MT5, análise de slippage e relatórios detalhados.

Author: Claude
Version: 2.0
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Optional imports for visualization and MT5
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Visualizations disabled.")

try:
    import MetaTrader5 as mt5
    HAS_MT5 = True
except ImportError:
    HAS_MT5 = False
    print("Warning: MetaTrader5 not installed. Direct MT5 connection disabled.")

try:
    from dotenv import load_dotenv
    load_dotenv()
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class ComparisonConfig:
    """Configuration for trade comparison."""
    # Tolerância para matching de preços (em pontos)
    price_tolerance: float = 30.0
    
    # Tolerância para matching de tempo (em segundos)
    time_tolerance_seconds: int = 30
    
    # Valor do ponto (para cálculo de diferenças em R$)
    point_value: float = 0.20
    
    # Timezone do MT5 (ajustar conforme necessário)
    mt5_timezone: str = 'America/Sao_Paulo'
    
    # Configurações de output
    output_dir: str = './reports'
    generate_plots: bool = True
    generate_excel: bool = True
    generate_txt: bool = True


# =============================================================================
# MT5 CONNECTION
# =============================================================================
def get_mt5_credentials() -> Dict:
    """
    Carrega credenciais do MT5 do arquivo .env ou variáveis de ambiente.
    
    Returns
    -------
    dict
        Dicionário com login, password, server e path do MT5
    """
    return {
        'login': int(os.getenv('MT5_LOGIN', 0)),
        'password': os.getenv('MT5_PASSWORD', ''),
        'server': os.getenv('MT5_SERVER', ''),
        'path': os.getenv('MT5_PATH', r"C:\Program Files\MetaTrader 5\terminal64.exe")
    }


def load_config(config_file: str) -> Optional[Dict]:
    """
    Carrega todas as configurações do arquivo JSON.
    
    Parameters
    ----------
    config_file : str
        Caminho para o arquivo de configuração JSON
    
    Returns
    -------
    dict or None
        Configurações ou None se falhar
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Arquivo {config_file} não encontrado!")
        return None
    except json.JSONDecodeError:
        print(f"Erro ao decodificar JSON do arquivo {config_file}")
        return None
    except Exception as e:
        print(f"Erro ao carregar {config_file}: {e}")
        return None


def connect_mt5(config: Dict = None) -> bool:
    """
    Estabelece conexão com MT5.
    
    Parameters
    ----------
    config : dict, optional
        Configurações de conexão. Se None, carrega do .env
    
    Returns
    -------
    bool
        True se conectado com sucesso
    """
    if not HAS_MT5:
        print("Erro: MetaTrader5 não instalado. Execute: pip install MetaTrader5")
        return False
    
    if config is None:
        config = get_mt5_credentials()
    
    # Validar credenciais
    if not config['login'] or not config['password'] or not config['server']:
        print("Erro: Credenciais do MT5 não encontradas no arquivo .env")
        print("Certifique-se de que MT5_LOGIN, MT5_PASSWORD e MT5_SERVER estão definidos")
        return False
    
    if not mt5.initialize(
        login=config['login'], 
        server=config['server'], 
        password=config['password'], 
        path=config['path']
    ):
        print("initialize() failed, error code =", mt5.last_error())
        return False
    
    print('Ligado ao MT5 com sucesso!')
    print(f'Conta: {config["login"]} | Servidor: {config["server"]}')
    print('-' * 50)
    return True


def disconnect_mt5():
    """Desconecta do MT5."""
    if HAS_MT5:
        mt5.shutdown()
        print("Desconectado do MT5.")


def get_mt5_deals(
    symbol: str = None,
    start_date: datetime = None,
    end_date: datetime = None,
    magic_number: int = None,
    consolidate: bool = True
) -> pd.DataFrame:
    """
    Obtém histórico de deals do MT5.
    
    Parameters
    ----------
    symbol : str, optional
        Símbolo para filtrar (ex: "WIN", "WING26")
    start_date : datetime, optional
        Data inicial (default: 30 dias atrás)
    end_date : datetime, optional
        Data final (default: agora)
    magic_number : int, optional
        Magic number do EA para filtrar
    consolidate : bool, default True
        Se True, consolida deals de entrada e saída em trades únicos
    
    Returns
    -------
    pd.DataFrame
        DataFrame com trades do MT5 (consolidados por position_id)
    """
    if not HAS_MT5:
        raise ImportError("MetaTrader5 não instalado")
    
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    if end_date is None:
        end_date = datetime.now()
    
    # Buscar deals
    deals = mt5.history_deals_get(start_date, end_date)
    
    if deals is None or len(deals) == 0:
        print("Nenhum deal encontrado no período")
        return pd.DataFrame()
    
    # Converter para DataFrame
    df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
    
    print(f"MT5 Terminal: {len(df)} deals brutos carregados")
    
    # Filtrar por símbolo
    if symbol:
        symbol_clean = symbol.replace('$', '')
        df = df[df['symbol'].str.contains(symbol_clean, case=False, na=False)]
    
    # Filtrar por magic number
    if magic_number:
        df = df[df['magic'] == magic_number]
    
    # Filtrar apenas deals de trade (excluir balance, etc)
    # entry: 0 = DEAL_ENTRY_IN, 1 = DEAL_ENTRY_OUT
    if 'entry' in df.columns:
        df = df[df['entry'].isin([0, 1])]
    
    if len(df) == 0:
        print("Nenhum deal de trade encontrado após filtros")
        return pd.DataFrame()
    
    if consolidate:
        df = _consolidate_mt5_deals(df)
        print(f"MT5 Terminal: {len(df)} trades consolidados")
    else:
        # Padronizar colunas sem consolidar
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df['entry_price'] = df['price']
        df['direction'] = np.where(df['type'] == 0, 'BUY', 'SELL')
    
    if len(df) > 0:
        print(f"  Período: {df['timestamp'].min()} a {df['timestamp'].max()}")
    
    return df


def _consolidate_mt5_deals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolida deals de entrada e saída em trades únicos usando position_id.
    
    No MT5, cada trade tem dois deals:
    - Deal de entrada (entry=0, DEAL_ENTRY_IN)
    - Deal de saída (entry=1, DEAL_ENTRY_OUT)
    
    Esta função agrupa por position_id e cria um registro único por trade.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com deals brutos do MT5
    
    Returns
    -------
    pd.DataFrame
        DataFrame com trades consolidados
    """
    if 'position_id' not in df.columns:
        print("Warning: position_id não encontrado. Retornando deals não consolidados.")
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df['entry_price'] = df['price']
        df['direction'] = np.where(df['type'] == 0, 'BUY', 'SELL')
        return df
    
    # Separar deals de entrada e saída
    # entry: 0 = IN (abertura), 1 = OUT (fechamento)
    entries = df[df['entry'] == 0].copy()
    exits = df[df['entry'] == 1].copy()
    
    if len(entries) == 0:
        print("Warning: Nenhum deal de entrada encontrado")
        return pd.DataFrame()
    
    # Preparar colunas de entrada
    entries = entries.rename(columns={
        'time': 'entry_time',
        'price': 'entry_price',
        'volume': 'entry_volume'
    })
    entries['entry_time'] = pd.to_datetime(entries['entry_time'], unit='s')
    
    # Preparar colunas de saída
    exits = exits.rename(columns={
        'time': 'exit_time',
        'price': 'exit_price',
        'volume': 'exit_volume',
        'profit': 'profit'
    })
    exits['exit_time'] = pd.to_datetime(exits['exit_time'], unit='s')
    
    # Colunas para manter de cada lado
    entry_cols = ['position_id', 'entry_time', 'entry_price', 'entry_volume', 
                  'type', 'symbol', 'magic', 'ticket']
    exit_cols = ['position_id', 'exit_time', 'exit_price', 'exit_volume', 
                 'profit', 'commission', 'swap', 'fee']
    
    # Filtrar apenas colunas existentes
    entry_cols = [c for c in entry_cols if c in entries.columns]
    exit_cols = [c for c in exit_cols if c in exits.columns]
    
    entries_clean = entries[entry_cols].copy()
    exits_clean = exits[[c for c in exit_cols if c in exits.columns]].copy()
    
    # Merge por position_id
    trades = pd.merge(
        entries_clean, 
        exits_clean, 
        on='position_id', 
        how='left'  # Manter entradas mesmo sem saída correspondente
    )
    
    # Padronizar colunas
    trades['timestamp'] = trades['entry_time']
    
    # Direção: type 0 = BUY, type 1 = SELL
    if 'type' in trades.columns:
        trades['direction'] = np.where(trades['type'] == 0, 'BUY', 'SELL')
    
    # Calcular pontos se não tiver profit
    if 'profit' not in trades.columns or trades['profit'].isna().all():
        if 'exit_price' in trades.columns and 'entry_price' in trades.columns:
            # Para BUY: exit - entry, para SELL: entry - exit
            trades['points'] = np.where(
                trades['direction'] == 'BUY',
                trades['exit_price'] - trades['entry_price'],
                trades['entry_price'] - trades['exit_price']
            )
    
    # Remover trades sem saída (posições ainda abertas)
    if 'exit_time' in trades.columns:
        open_positions = trades['exit_time'].isna().sum()
        if open_positions > 0:
            print(f"  Info: {open_positions} posições ainda abertas (ignoradas)")
            trades = trades[trades['exit_time'].notna()]
    
    # Reordenar colunas
    col_order = ['timestamp', 'position_id', 'symbol', 'direction', 'entry_price', 
                 'exit_price', 'entry_time', 'exit_time', 'profit', 'magic']
    col_order = [c for c in col_order if c in trades.columns]
    other_cols = [c for c in trades.columns if c not in col_order]
    trades = trades[col_order + other_cols]
    
    return trades.reset_index(drop=True)


def get_mt5_positions_history(
    symbol: str = None,
    start_date: datetime = None,
    end_date: datetime = None,
    magic_number: int = None
) -> pd.DataFrame:
    """
    Obtém histórico de posições fechadas do MT5 (agrupando entrada e saída).
    
    Parameters
    ----------
    symbol : str, optional
        Símbolo para filtrar
    start_date : datetime, optional
        Data inicial
    end_date : datetime, optional
        Data final
    magic_number : int, optional
        Magic number do EA
    
    Returns
    -------
    pd.DataFrame
        DataFrame com posições fechadas
    """
    if not HAS_MT5:
        raise ImportError("MetaTrader5 não instalado")
    
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)
    if end_date is None:
        end_date = datetime.now()
    
    # Buscar orders do histórico
    orders = mt5.history_orders_get(start_date, end_date)
    
    if orders is None or len(orders) == 0:
        print("Nenhuma ordem encontrada no período")
        return pd.DataFrame()
    
    df = pd.DataFrame(list(orders), columns=orders[0]._asdict().keys())
    
    # Filtrar por símbolo
    if symbol:
        symbol_clean = symbol.replace('$', '')
        df = df[df['symbol'].str.contains(symbol_clean, case=False, na=False)]
    
    # Filtrar por magic number
    if magic_number:
        df = df[df['magic'] == magic_number]
    
    # Padronizar
    df['timestamp'] = pd.to_datetime(df['time_setup'], unit='s')
    df['entry_price'] = df['price_open']
    df['direction'] = np.where(df['type'] == 0, 'BUY', 'SELL')
    
    print(f"MT5 Terminal: {len(df)} orders carregadas")
    
    return df


def get_mt5_raw_deals(
    symbol: str = None,
    start_date: datetime = None,
    end_date: datetime = None,
    magic_number: int = None
) -> pd.DataFrame:
    """
    Obtém deals brutos do MT5 SEM consolidar (para debug).
    
    Cada trade terá 2 linhas: uma para entrada (entry=0) e uma para saída (entry=1).
    
    Parameters
    ----------
    symbol : str, optional
        Símbolo para filtrar
    start_date : datetime, optional
        Data inicial
    end_date : datetime, optional
        Data final
    magic_number : int, optional
        Magic number do EA
    
    Returns
    -------
    pd.DataFrame
        DataFrame com deals brutos (não consolidados)
    """
    return get_mt5_deals(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        magic_number=magic_number,
        consolidate=False  # Não consolidar
    )


# =============================================================================
# DATA LOADERS - FILES
# =============================================================================
def load_backtest_trades(path: str) -> pd.DataFrame:
    """
    Carrega trades do backtest Python.
    
    Parameters
    ----------
    path : str
        Caminho para arquivo CSV ou Parquet com trades do backtest
    
    Returns
    -------
    pd.DataFrame
        DataFrame padronizado com trades do backtest
    """
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    
    # Padronizar colunas
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Criar coluna de direção se não existir
    if 'direction' not in df.columns:
        if 'flag_buy' in df.columns and 'flag_sell' in df.columns:
            df['direction'] = np.where(df['flag_buy'] == 1, 'BUY', 'SELL')
        else:
            df['direction'] = 'UNKNOWN'
    
    # Renomear colunas para padrão
    column_mapping = {
        'trade_entry_price': 'entry_price',
        'trade_exit_price': 'exit_price',
        'trade_points': 'points',
        'trade_status': 'status'
    }
    
    for old, new in column_mapping.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    
    print(f"Backtest: {len(df)} trades carregados")
    print(f"  Período: {df['timestamp'].min()} a {df['timestamp'].max()}")
    
    return df


def load_mt5_trades_from_csv(path: str, consolidate: bool = True) -> pd.DataFrame:
    """
    Carrega trades exportados do MT5 (formato CSV do relatório).
    Suporta formato brasileiro com delimitador ; e preços com espaço.
    
    Parameters
    ----------
    path : str
        Caminho para arquivo CSV exportado do MT5
    consolidate : bool, default True
        Se True e detectar deals separados (entrada/saída), consolida em trades únicos
    
    Returns
    -------
    pd.DataFrame
        DataFrame padronizado com trades do MT5
    """
    # Tentar diferentes encodings e delimitadores
    encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
    delimiters = [';', ',', '\t']
    
    df = None
    for encoding in encodings:
        for delimiter in delimiters:
            try:
                df = pd.read_csv(path, encoding=encoding, sep=delimiter)
                # Verificar se tem colunas válidas (mais de 5 colunas)
                if len(df.columns) > 5:
                    break
            except:
                continue
        if df is not None and len(df.columns) > 5:
            break
    
    if df is None:
        raise ValueError(f"Não foi possível ler o arquivo: {path}")
    
    # Detectar e padronizar colunas do MT5
    df = _standardize_mt5_columns(df)
    
    # Detectar se precisa consolidar (tem position_id e entry separados)
    needs_consolidation = False
    if consolidate:
        # Verificar se parece ter deals duplicados (entrada/saída)
        if 'position_id' in df.columns or 'Position' in df.columns:
            pos_col = 'position_id' if 'position_id' in df.columns else 'Position'
            # Se position_id tem duplicatas, provavelmente precisa consolidar
            if df[pos_col].duplicated().any():
                needs_consolidation = True
                print(f"  Detectado formato com deals separados (entrada/saída)")
        
        # Verificar pelo número de linhas vs trades únicos
        if 'ticket' in df.columns:
            unique_tickets = df['ticket'].nunique()
            if len(df) > unique_tickets * 1.5:  # Mais de 1.5x = provavelmente duplicado
                needs_consolidation = True
    
    if needs_consolidation:
        df = _consolidate_csv_deals(df)
        print(f"MT5: {len(df)} trades consolidados")
    else:
        print(f"MT5: {len(df)} trades carregados")
    
    if 'timestamp' in df.columns and len(df) > 0:
        print(f"  Período: {df['timestamp'].min()} a {df['timestamp'].max()}")
    
    return df


def _consolidate_csv_deals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolida deals de entrada e saída em CSV quando estão em linhas separadas.
    
    Detecta automaticamente o formato e tenta agrupar por position_id ou ticket.
    """
    df = df.copy()
    
    # Tentar encontrar coluna de position/ticket para agrupar
    group_col = None
    for col in ['position_id', 'Position', 'ticket', 'Ticket', 'Deal']:
        if col in df.columns:
            group_col = col
            break
    
    if group_col is None:
        print("Warning: Não foi possível identificar coluna de agrupamento")
        return df
    
    # Se temos entrada e saída como colunas separadas (formato de relatório), retornar como está
    if 'entry_price' in df.columns and 'exit_price' in df.columns:
        if df['entry_price'].notna().any() and df['exit_price'].notna().any():
            return df
    
    # Agrupar por position/ticket
    trades = []
    
    for group_id, group_df in df.groupby(group_col):
        if len(group_df) == 1:
            # Trade único (já consolidado ou formato diferente)
            row = group_df.iloc[0].to_dict()
            trades.append(row)
        elif len(group_df) == 2:
            # Dois deals: entrada e saída
            # Ordenar por timestamp para identificar entrada (primeiro) e saída (segundo)
            group_df = group_df.sort_values('timestamp')
            entry_row = group_df.iloc[0]
            exit_row = group_df.iloc[1]
            
            trade = {
                'position_id': group_id,
                'timestamp': entry_row.get('timestamp'),
                'entry_time': entry_row.get('timestamp'),
                'exit_time': exit_row.get('timestamp'),
                'symbol': entry_row.get('symbol', exit_row.get('symbol')),
                'direction': entry_row.get('direction'),
                'entry_price': entry_row.get('entry_price', entry_row.get('price')),
                'exit_price': exit_row.get('exit_price', exit_row.get('price')),
                'volume': entry_row.get('volume', exit_row.get('volume')),
                'profit': exit_row.get('profit', 0),
                'commission': entry_row.get('commission', 0) + exit_row.get('commission', 0),
                'swap': exit_row.get('swap', 0),
            }
            trades.append(trade)
        else:
            # Mais de 2 deals para mesmo position_id (parciais?)
            # Pegar primeiro como entrada, último como saída
            group_df = group_df.sort_values('timestamp')
            entry_row = group_df.iloc[0]
            exit_row = group_df.iloc[-1]
            
            trade = {
                'position_id': group_id,
                'timestamp': entry_row.get('timestamp'),
                'entry_time': entry_row.get('timestamp'),
                'exit_time': exit_row.get('timestamp'),
                'symbol': entry_row.get('symbol', exit_row.get('symbol')),
                'direction': entry_row.get('direction'),
                'entry_price': entry_row.get('entry_price', entry_row.get('price')),
                'exit_price': exit_row.get('exit_price', exit_row.get('price')),
                'volume': group_df['volume'].sum() if 'volume' in group_df else None,
                'profit': group_df['profit'].sum() if 'profit' in group_df else 0,
            }
            trades.append(trade)
    
    result = pd.DataFrame(trades)
    
    # Garantir que timestamp está no formato correto
    if 'timestamp' in result.columns:
        result['timestamp'] = pd.to_datetime(result['timestamp'])
    
    return result


def load_mt5_trades_from_html(path: str) -> pd.DataFrame:
    """
    Carrega trades do relatório HTML do MT5.
    
    Parameters
    ----------
    path : str
        Caminho para arquivo HTML exportado do MT5
    
    Returns
    -------
    pd.DataFrame
        DataFrame padronizado com trades do MT5
    """
    tables = pd.read_html(path)
    
    # MT5 geralmente coloca os trades na maior tabela
    df = max(tables, key=len)
    
    df = _standardize_mt5_columns(df)
    
    print(f"MT5 (HTML): {len(df)} trades carregados")
    
    return df


def _clean_price(x) -> float:
    """Limpa preço do formato brasileiro (com espaço) para float."""
    if pd.isna(x):
        return np.nan
    return float(str(x).replace(' ', '').replace(',', '.'))


def _clean_profit(x) -> float:
    """Limpa valor de lucro do formato brasileiro para float."""
    if pd.isna(x):
        return 0
    s = str(x).replace(' ', '').replace(',', '.')
    try:
        return float(s)
    except:
        return 0


def _standardize_mt5_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza colunas do MT5 para formato comum.
    Suporta formato brasileiro com delimitador ; e preços com espaço.
    """
    df = df.copy()
    
    # Detectar se é formato brasileiro (colunas com acento)
    cols_lower = [str(c).lower() for c in df.columns]
    is_brazilian = any('horário' in c or 'preço' in c or 'lucro' in c for c in cols_lower)
    
    if is_brazilian:
        # Formato brasileiro: Horário;Position;Ativo;Tipo;Volume;Preço;S / L;T / P;Horário.1;Preço.1;Comissão;Swap;Lucro
        # Mapear por índice de coluna
        col_names = df.columns.tolist()
        
        if len(col_names) >= 13:
            # Timestamp de entrada (primeira coluna Horário)
            df['timestamp'] = pd.to_datetime(df.iloc[:, 0], format='%Y.%m.%d %H:%M:%S', errors='coerce')
            
            # Preço de entrada (coluna 5, índice 5)
            df['entry_price'] = df.iloc[:, 5].apply(_clean_price)
            
            # Timestamp de saída (coluna Horário.1, índice 8)
            if len(col_names) > 8:
                df['exit_time'] = pd.to_datetime(df.iloc[:, 8], format='%Y.%m.%d %H:%M:%S', errors='coerce')
            
            # Preço de saída (coluna Preço.1, índice 9)
            if len(col_names) > 9:
                df['exit_price'] = df.iloc[:, 9].apply(_clean_price)
            
            # Direção (coluna Tipo, índice 3)
            df['direction'] = df.iloc[:, 3].astype(str).str.upper()
            
            # Volume (índice 4)
            df['volume'] = pd.to_numeric(df.iloc[:, 4], errors='coerce')
            
            # S/L e T/P (índices 6 e 7)
            df['sl'] = df.iloc[:, 6].apply(_clean_price)
            df['tp'] = df.iloc[:, 7].apply(_clean_price)
            
            # Lucro (índice 12)
            if len(col_names) > 12:
                df['profit'] = df.iloc[:, 12].apply(_clean_profit)
            
            # Ticket/Position (índice 1)
            df['ticket'] = df.iloc[:, 1]
            
            # Símbolo (índice 2)
            df['symbol'] = df.iloc[:, 2]
    
    else:
        # Formato padrão internacional
        column_mappings = {
            'timestamp': ['Time', 'Hora', 'Open Time', 'Hora Abertura', 'Data/Hora', 
                          'Open time', 'Entry time', 'Tempo'],
            'entry_price': ['Price', 'Preço', 'Open Price', 'Preço Abertura', 
                            'Entry Price', 'Preço Entrada', 'Open price'],
            'exit_price': ['Close Price', 'Preço Fechamento', 'Exit Price', 
                           'Preço Saída', 'Close price'],
            'direction': ['Type', 'Tipo', 'Direction', 'Direção', 'Side'],
            'volume': ['Volume', 'Lots', 'Lotes', 'Size', 'Tamanho'],
            'profit': ['Profit', 'Lucro', 'P/L', 'Result', 'Resultado'],
            'symbol': ['Symbol', 'Símbolo', 'Ativo'],
            'ticket': ['Ticket', 'Order', 'Ordem', 'Deal', 'Negócio', '#'],
            'sl': ['S/L', 'Stop Loss', 'SL', 'S / L'],
            'tp': ['T/P', 'Take Profit', 'TP', 'T / P'],
            'exit_time': ['Close Time', 'Hora Fechamento', 'Exit Time', 'Close time']
        }
        
        for standard_name, possible_names in column_mappings.items():
            for col in df.columns:
                if col in possible_names or str(col).lower() in [p.lower() for p in possible_names]:
                    df[standard_name] = df[col]
                    break
        
        # Converter timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        if 'exit_time' in df.columns:
            df['exit_time'] = pd.to_datetime(df['exit_time'], errors='coerce')
        
        # Converter colunas numéricas
        numeric_cols = ['entry_price', 'exit_price', 'volume', 'profit', 'sl', 'tp']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].apply(_clean_price)
    
    # Padronizar direção
    if 'direction' in df.columns:
        df['direction'] = df['direction'].astype(str).str.upper()
        df['direction'] = df['direction'].replace({
            'BUY': 'BUY', 'COMPRA': 'BUY', 'LONG': 'BUY', 'C': 'BUY',
            'SELL': 'SELL', 'VENDA': 'SELL', 'SHORT': 'SELL', 'V': 'SELL'
        })
    
    return df


# =============================================================================
# TRADE MATCHING
# =============================================================================
def match_trades(
    backtest_trades: pd.DataFrame,
    mt5_trades: pd.DataFrame,
    config: ComparisonConfig = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Faz o matching entre trades do backtest e do MT5.
    
    Parameters
    ----------
    backtest_trades : pd.DataFrame
        Trades do backtest
    mt5_trades : pd.DataFrame
        Trades do MT5
    config : ComparisonConfig
        Configurações de tolerância
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (matched_trades, unmatched_backtest, unmatched_mt5)
    """
    if config is None:
        config = ComparisonConfig()
    
    bt = backtest_trades.copy()
    mt5 = mt5_trades.copy()
    
    # Garantir que temos as colunas necessárias
    required_bt = ['timestamp', 'direction', 'entry_price']
    required_mt5 = ['timestamp', 'direction', 'entry_price']
    
    for col in required_bt:
        if col not in bt.columns:
            raise ValueError(f"Coluna '{col}' não encontrada no backtest")
    
    for col in required_mt5:
        if col not in mt5.columns:
            raise ValueError(f"Coluna '{col}' não encontrada no MT5")
    
    # Adicionar índices originais
    bt['bt_idx'] = bt.index
    mt5['mt5_idx'] = mt5.index
    
    matched = []
    used_mt5_indices = set()
    
    for _, bt_row in bt.iterrows():
        best_match = None
        best_score = float('inf')
        
        bt_time = bt_row['timestamp']
        bt_entry = bt_row['entry_price']
        bt_dir = bt_row['direction']
        
        for _, mt5_row in mt5.iterrows():
            if mt5_row['mt5_idx'] in used_mt5_indices:
                continue
            
            # Diferença de tempo
            time_diff = abs((mt5_row['timestamp'] - bt_time).total_seconds())
            
            # Diferença de preço
            price_diff = abs(mt5_row['entry_price'] - bt_entry)
            
            # Verificar tolerâncias
            if time_diff > config.time_tolerance_seconds:
                continue
            if price_diff > config.price_tolerance:
                continue
            
            # Calcular score (menor = melhor)
            dir_match = 1 if mt5_row['direction'] == bt_dir else 0
            score = time_diff + price_diff * 0.1 - dir_match * 10
            
            if score < best_score:
                best_score = score
                best_match = mt5_row
        
        if best_match is not None:
            used_mt5_indices.add(best_match['mt5_idx'])
            
            matched.append({
                'bt_idx': bt_row['bt_idx'],
                'mt5_idx': best_match['mt5_idx'],
                'bt_timestamp': bt_row['timestamp'],
                'mt5_timestamp': best_match['timestamp'],
                'time_diff_sec': (best_match['timestamp'] - bt_row['timestamp']).total_seconds(),
                'bt_direction': bt_row['direction'],
                'mt5_direction': best_match['direction'],
                'direction_match': bt_row['direction'] == best_match['direction'],
                'bt_entry_price': bt_row['entry_price'],
                'mt5_entry_price': best_match['entry_price'],
                'entry_diff': best_match['entry_price'] - bt_row['entry_price'],
                'bt_exit_price': bt_row.get('exit_price', np.nan),
                'mt5_exit_price': best_match.get('exit_price', np.nan),
                'exit_diff': best_match.get('exit_price', 0) - bt_row.get('exit_price', 0),
                'bt_points': bt_row.get('points', 0),
                'mt5_profit': best_match.get('profit', 0),
                'bt_status': bt_row.get('status', np.nan),
                'match_score': best_score
            })
    
    matched_df = pd.DataFrame(matched) if matched else pd.DataFrame()
    
    # Identificar não matcheados
    matched_bt_idx = set(matched_df['bt_idx']) if len(matched_df) > 0 else set()
    matched_mt5_idx = set(matched_df['mt5_idx']) if len(matched_df) > 0 else set()
    
    unmatched_bt = bt[~bt['bt_idx'].isin(matched_bt_idx)].copy()
    unmatched_mt5 = mt5[~mt5['mt5_idx'].isin(matched_mt5_idx)].copy()
    
    print(f"\nMatching concluído:")
    print(f"  Matcheados: {len(matched_df)}")
    print(f"  Não matcheados (BT): {len(unmatched_bt)}")
    print(f"  Não matcheados (MT5): {len(unmatched_mt5)}")
    
    return matched_df, unmatched_bt, unmatched_mt5


# =============================================================================
# ANALYSIS
# =============================================================================
def analyze_comparison(
    matched: pd.DataFrame,
    unmatched_bt: pd.DataFrame,
    unmatched_mt5: pd.DataFrame,
    config: ComparisonConfig = None
) -> Dict:
    """
    Analisa os resultados da comparação.
    
    Returns
    -------
    dict
        Dicionário com métricas de comparação
    """
    if config is None:
        config = ComparisonConfig()
    
    results = {
        'total_backtest_trades': len(matched) + len(unmatched_bt),
        'total_mt5_trades': len(matched) + len(unmatched_mt5),
        'matched_trades': len(matched),
        'unmatched_backtest': len(unmatched_bt),
        'unmatched_mt5': len(unmatched_mt5),
        'match_rate_backtest': 0,
        'match_rate_mt5': 0,
    }
    
    if results['total_backtest_trades'] > 0:
        results['match_rate_backtest'] = len(matched) / results['total_backtest_trades'] * 100
    
    if results['total_mt5_trades'] > 0:
        results['match_rate_mt5'] = len(matched) / results['total_mt5_trades'] * 100
    
    if len(matched) > 0:
        # Análise de diferenças nos trades matcheados
        results['avg_entry_diff'] = matched['entry_diff'].mean()
        results['avg_exit_diff'] = matched['exit_diff'].mean()
        results['max_entry_diff'] = matched['entry_diff'].abs().max()
        results['max_exit_diff'] = matched['exit_diff'].abs().max()
        results['std_entry_diff'] = matched['entry_diff'].std()
        results['std_exit_diff'] = matched['exit_diff'].std()
        
        # Slippage absoluto
        results['avg_entry_slippage'] = matched['entry_diff'].abs().mean()
        results['avg_exit_slippage'] = matched['exit_diff'].abs().mean()
        
        # Diferença de tempo
        results['avg_time_diff_seconds'] = matched['time_diff_sec'].abs().mean()
        results['max_time_diff_seconds'] = matched['time_diff_sec'].abs().max()
        
        # Comparar resultados
        results['bt_total_points'] = matched['bt_points'].sum()
        results['mt5_total_profit'] = matched['mt5_profit'].sum()
        results['bt_total_profit_brl'] = matched['bt_points'].sum() * config.point_value
        
        # Direções corretas
        results['direction_match_rate'] = matched['direction_match'].mean() * 100
        
        # Win rates
        matched['bt_win'] = matched['bt_points'] > 0
        matched['mt5_win'] = matched['mt5_profit'] > 0
        matched['outcome_match'] = matched['bt_win'] == matched['mt5_win']
        
        results['bt_win_rate'] = matched['bt_win'].mean() * 100
        results['mt5_win_rate'] = matched['mt5_win'].mean() * 100
        results['bt_wins'] = matched['bt_win'].sum()
        results['mt5_wins'] = matched['mt5_win'].sum()
        
        # Divergências de resultado
        results['outcome_match_rate'] = matched['outcome_match'].mean() * 100
        results['divergent_trades'] = (~matched['outcome_match']).sum()
        results['bt_win_mt5_loss'] = (matched['bt_win'] & ~matched['mt5_win']).sum()
        results['bt_loss_mt5_win'] = (~matched['bt_win'] & matched['mt5_win']).sum()
        
        # Análise por hora
        matched['hour'] = pd.to_datetime(matched['bt_timestamp']).dt.hour
        divergent_by_hour = matched[~matched['outcome_match']].groupby('hour').size()
        total_by_hour = matched.groupby('hour').size()
        results['divergent_rate_by_hour'] = (divergent_by_hour / total_by_hour * 100).to_dict()
    
    return results


def analyze_slippage(matched: pd.DataFrame, config: ComparisonConfig = None) -> Dict:
    """
    Análise detalhada de slippage.
    
    Returns
    -------
    dict
        Métricas de slippage
    """
    if config is None:
        config = ComparisonConfig()
    
    if len(matched) == 0:
        return {}
    
    results = {}
    
    # Entrada
    entry_diff = matched['entry_diff']
    results['entry'] = {
        'mean': entry_diff.mean(),
        'std': entry_diff.std(),
        'min': entry_diff.min(),
        'max': entry_diff.max(),
        'abs_mean': entry_diff.abs().mean(),
        'median': entry_diff.median(),
        'positive_count': (entry_diff > 0).sum(),  # MT5 preço maior
        'negative_count': (entry_diff < 0).sum(),  # MT5 preço menor
    }
    
    # Saída
    exit_diff = matched['exit_diff']
    results['exit'] = {
        'mean': exit_diff.mean(),
        'std': exit_diff.std(),
        'min': exit_diff.min(),
        'max': exit_diff.max(),
        'abs_mean': exit_diff.abs().mean(),
        'median': exit_diff.median(),
        'positive_count': (exit_diff > 0).sum(),
        'negative_count': (exit_diff < 0).sum(),
    }
    
    # Tempo
    time_diff = matched['time_diff_sec']
    results['time'] = {
        'mean': time_diff.abs().mean(),
        'std': time_diff.std(),
        'min': time_diff.min(),
        'max': time_diff.max(),
        'median': time_diff.median(),
    }
    
    # Impacto financeiro do slippage
    # Slippage na entrada afeta a direção do trade
    # Slippage na saída afeta o resultado
    matched['slippage_impact'] = matched['exit_diff'] * np.where(
        matched['bt_direction'] == 'BUY', -1, 1
    ) * config.point_value
    
    results['financial_impact'] = {
        'total': matched['slippage_impact'].sum(),
        'mean': matched['slippage_impact'].mean(),
    }
    
    return results


# =============================================================================
# REPORTING - TEXT
# =============================================================================
def print_comparison_report(
    matched: pd.DataFrame,
    unmatched_bt: pd.DataFrame,
    unmatched_mt5: pd.DataFrame,
    analysis: Dict,
    config: ComparisonConfig = None
) -> str:
    """
    Gera e imprime relatório detalhado da comparação.
    
    Returns
    -------
    str
        Texto do relatório
    """
    if config is None:
        config = ComparisonConfig()
    
    lines = []
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("RELATÓRIO DE COMPARAÇÃO: BACKTEST vs MT5")
    lines.append("=" * 80)
    
    lines.append("\n📊 RESUMO GERAL")
    lines.append("-" * 50)
    lines.append(f"Trades no Backtest:     {analysis['total_backtest_trades']}")
    lines.append(f"Trades no MT5:          {analysis['total_mt5_trades']}")
    lines.append(f"Trades Matcheados:      {analysis['matched_trades']}")
    lines.append(f"Não matcheados (BT):    {analysis['unmatched_backtest']}")
    lines.append(f"Não matcheados (MT5):   {analysis['unmatched_mt5']}")
    lines.append(f"Taxa de Match (BT):     {analysis['match_rate_backtest']:.1f}%")
    lines.append(f"Taxa de Match (MT5):    {analysis['match_rate_mt5']:.1f}%")
    
    if analysis['matched_trades'] > 0:
        lines.append("\n📈 ANÁLISE DOS TRADES MATCHEADOS")
        lines.append("-" * 50)
        lines.append(f"Match de direção:       {analysis.get('direction_match_rate', 0):.1f}%")
        lines.append(f"Slippage médio entrada: {analysis.get('avg_entry_slippage', 0):.1f} pts")
        lines.append(f"Slippage médio saída:   {analysis.get('avg_exit_slippage', 0):.1f} pts")
        lines.append(f"Latência média:         {analysis.get('avg_time_diff_seconds', 0):.1f}s")
        
        lines.append("\n💰 COMPARAÇÃO DE RESULTADOS")
        lines.append("-" * 50)
        lines.append(f"Total pontos (BT):      {analysis.get('bt_total_points', 0):.0f}")
        lines.append(f"Total profit BT (R$):   {analysis.get('bt_total_profit_brl', 0):.2f}")
        lines.append(f"Total profit MT5 (R$):  {analysis.get('mt5_total_profit', 0):.2f}")
        diff = analysis.get('bt_total_profit_brl', 0) - analysis.get('mt5_total_profit', 0)
        lines.append(f"Diferença (R$):         {diff:.2f}")
        
        lines.append("\n📊 WIN RATES")
        lines.append("-" * 50)
        lines.append(f"Win Rate Backtest:      {analysis.get('bt_win_rate', 0):.1f}% ({analysis.get('bt_wins', 0)} trades)")
        lines.append(f"Win Rate MT5:           {analysis.get('mt5_win_rate', 0):.1f}% ({analysis.get('mt5_wins', 0)} trades)")
        
        lines.append("\n⚠️  DIVERGÊNCIAS DE RESULTADO")
        lines.append("-" * 50)
        lines.append(f"Trades com resultado igual:    {analysis.get('outcome_match_rate', 0):.1f}%")
        lines.append(f"Trades divergentes:            {analysis.get('divergent_trades', 0)}")
        lines.append(f"  BT ganhou, MT5 perdeu:       {analysis.get('bt_win_mt5_loss', 0)}")
        lines.append(f"  BT perdeu, MT5 ganhou:       {analysis.get('bt_loss_mt5_win', 0)}")
    
    # Trades não matcheados
    if len(unmatched_bt) > 0:
        lines.append("\n📍 TRADES DO BACKTEST NÃO ENCONTRADOS NO MT5 (primeiros 10)")
        lines.append("-" * 50)
        cols_to_show = ['timestamp', 'direction', 'entry_price', 'exit_price', 'points']
        cols_available = [c for c in cols_to_show if c in unmatched_bt.columns]
        for _, row in unmatched_bt.head(10).iterrows():
            line = f"  {row.get('timestamp', 'N/A')} | {row.get('direction', 'N/A')} | "
            line += f"Entry: {row.get('entry_price', 0):.0f} | Pts: {row.get('points', 0):.0f}"
            lines.append(line)
        if len(unmatched_bt) > 10:
            lines.append(f"  ... e mais {len(unmatched_bt) - 10} trades")
    
    if len(unmatched_mt5) > 0:
        lines.append("\n📍 TRADES DO MT5 NÃO ENCONTRADOS NO BACKTEST (primeiros 10)")
        lines.append("-" * 50)
        for _, row in unmatched_mt5.head(10).iterrows():
            line = f"  {row.get('timestamp', 'N/A')} | {row.get('direction', 'N/A')} | "
            line += f"Entry: {row.get('entry_price', 0):.0f} | Profit: {row.get('profit', 0):.2f}"
            lines.append(line)
        if len(unmatched_mt5) > 10:
            lines.append(f"  ... e mais {len(unmatched_mt5) - 10} trades")
    
    lines.append("\n" + "=" * 80)
    
    # Diagnóstico
    lines.append("\n🔍 DIAGNÓSTICO")
    lines.append("-" * 50)
    
    if analysis['match_rate_backtest'] >= 95:
        lines.append("✅ Excelente correspondência entre backtest e MT5!")
    elif analysis['match_rate_backtest'] >= 80:
        lines.append("⚠️  Boa correspondência, mas há algumas discrepâncias.")
    elif analysis['match_rate_backtest'] >= 50:
        lines.append("⚠️  Correspondência moderada. Possíveis causas:")
        lines.append("   - Diferença de timezone")
        lines.append("   - Slippage significativo")
        lines.append("   - Configurações diferentes (SL/TP/distância)")
    else:
        lines.append("❌ Baixa correspondência. Verifique:")
        lines.append("   - Se os parâmetros do EA estão iguais ao backtest")
        lines.append("   - Se o período de dados é o mesmo")
        lines.append("   - Se há problemas de conexão/execução")
    
    if analysis.get('avg_exit_slippage', 0) > 10:
        lines.append("")
        lines.append("⚠️  ALERTA: Slippage na saída alto!")
        lines.append(f"   Média de {analysis.get('avg_exit_slippage', 0):.1f} pts pode inverter resultados")
        lines.append("   com TP/SL apertados. Considere aumentar TP/SL.")
    
    lines.append("\n" + "=" * 80)
    
    report = '\n'.join(lines)
    print(report)
    
    return report


# =============================================================================
# REPORTING - EXCEL
# =============================================================================
def export_to_excel(
    matched: pd.DataFrame,
    unmatched_bt: pd.DataFrame,
    unmatched_mt5: pd.DataFrame,
    analysis: Dict,
    output_path: str,
    config: ComparisonConfig = None
) -> None:
    """
    Exporta relatório de comparação para Excel.
    """
    if config is None:
        config = ComparisonConfig()
    
    try:
        import openpyxl
    except ImportError:
        print("openpyxl não instalado. Execute: pip install openpyxl")
        return
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Resumo
        summary_data = {
            'Métrica': [
                'Total Trades Backtest', 'Total Trades MT5', 'Trades Matcheados',
                'Taxa de Match (%)', 'Win Rate Backtest (%)', 'Win Rate MT5 (%)',
                'Profit Backtest (R$)', 'Profit MT5 (R$)', 'Diferença (R$)',
                'Slippage Entrada (pts)', 'Slippage Saída (pts)', 'Latência (s)',
                'Trades Divergentes', '% Divergentes',
                'BT Ganhou/MT5 Perdeu', 'BT Perdeu/MT5 Ganhou'
            ],
            'Valor': [
                analysis['total_backtest_trades'],
                analysis['total_mt5_trades'],
                analysis['matched_trades'],
                f"{analysis['match_rate_backtest']:.1f}",
                f"{analysis.get('bt_win_rate', 0):.1f}",
                f"{analysis.get('mt5_win_rate', 0):.1f}",
                f"{analysis.get('bt_total_profit_brl', 0):.2f}",
                f"{analysis.get('mt5_total_profit', 0):.2f}",
                f"{analysis.get('bt_total_profit_brl', 0) - analysis.get('mt5_total_profit', 0):.2f}",
                f"{analysis.get('avg_entry_slippage', 0):.1f}",
                f"{analysis.get('avg_exit_slippage', 0):.1f}",
                f"{analysis.get('avg_time_diff_seconds', 0):.1f}",
                analysis.get('divergent_trades', 0),
                f"{100 - analysis.get('outcome_match_rate', 100):.1f}",
                analysis.get('bt_win_mt5_loss', 0),
                analysis.get('bt_loss_mt5_win', 0),
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Resumo', index=False)
        
        # Trades matcheados
        if len(matched) > 0:
            matched_export = matched[[c for c in matched.columns if c not in ['bt_win', 'mt5_win', 'outcome_match', 'hour']]].copy()
            matched_export.to_excel(writer, sheet_name='Trades_Matcheados', index=False)
            
            # Trades divergentes
            if 'outcome_match' in matched.columns:
                divergent = matched[~matched['outcome_match']].copy()
                if len(divergent) > 0:
                    divergent.to_excel(writer, sheet_name='Trades_Divergentes', index=False)
        
        # Não matcheados
        if len(unmatched_bt) > 0:
            unmatched_bt.to_excel(writer, sheet_name='Nao_Match_Backtest', index=False)
        
        if len(unmatched_mt5) > 0:
            cols = ['timestamp', 'direction', 'entry_price', 'exit_price', 'profit']
            cols_available = [c for c in cols if c in unmatched_mt5.columns]
            unmatched_mt5[cols_available].to_excel(writer, sheet_name='Nao_Match_MT5', index=False)
    
    print(f"📁 Relatório Excel salvo: {output_path}")


# =============================================================================
# REPORTING - VISUALIZATION
# =============================================================================
def generate_comparison_plots(
    matched: pd.DataFrame,
    analysis: Dict,
    output_path: str,
    config: ComparisonConfig = None
) -> None:
    """
    Gera gráficos de comparação.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib não instalado. Gráficos desabilitados.")
        return
    
    if config is None:
        config = ComparisonConfig()
    
    if len(matched) == 0:
        print("Sem trades matcheados para plotar.")
        return
    
    # Preparar dados
    matched = matched.copy()
    matched['bt_timestamp'] = pd.to_datetime(matched['bt_timestamp'])
    matched['bt_win'] = matched['bt_points'] > 0
    matched['mt5_win'] = matched['mt5_profit'] > 0
    matched['outcome_match'] = matched['bt_win'] == matched['mt5_win']
    
    # Configurar estilo
    #plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Criar figura
    fig = plt.figure(figsize=(16, 20))
    gs = GridSpec(5, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    # Cores
    C_BT = '#2196F3'
    C_MT5 = '#FF9800'
    C_WIN = '#4CAF50'
    C_LOSS = '#F44336'
    
    # === 1. EQUITY CURVES ===
    ax1 = fig.add_subplot(gs[0, :])
    bt_cumsum = (matched['bt_points'] * config.point_value).cumsum()
    mt5_cumsum = matched['mt5_profit'].cumsum()
    
    ax1.plot(range(len(bt_cumsum)), bt_cumsum, color=C_BT, linewidth=2.5, label='Backtest')
    ax1.plot(range(len(mt5_cumsum)), mt5_cumsum, color=C_MT5, linewidth=2.5, label='MT5')
    ax1.fill_between(range(len(bt_cumsum)), 0, bt_cumsum, alpha=0.15, color=C_BT)
    ax1.fill_between(range(len(mt5_cumsum)), 0, mt5_cumsum, alpha=0.15, color=C_MT5)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Numero do Trade', fontsize=11)
    ax1.set_ylabel('Profit Acumulado (R$)', fontsize=11)
    ax1.set_title('CURVAS DE EQUITY: Backtest vs MT5', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11)
    ax1.annotate(f'R$ {bt_cumsum.iloc[-1]:.0f}', xy=(len(bt_cumsum)-1, bt_cumsum.iloc[-1]), 
                 xytext=(10, 5), textcoords='offset points', color=C_BT, fontweight='bold', fontsize=11)
    ax1.annotate(f'R$ {mt5_cumsum.iloc[-1]:.0f}', xy=(len(mt5_cumsum)-1, mt5_cumsum.iloc[-1]), 
                 xytext=(10, -5), textcoords='offset points', color=C_MT5, fontweight='bold', fontsize=11)
    
    # === 2. WIN RATE COMPARISON ===
    ax2 = fig.add_subplot(gs[1, 0])
    categories = ['Backtest', 'MT5']
    wins = [matched['bt_win'].sum(), matched['mt5_win'].sum()]
    losses = [len(matched) - wins[0], len(matched) - wins[1]]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax2.bar(x - width/2, wins, width, label='Ganhos', color=C_WIN, alpha=0.85)
    ax2.bar(x + width/2, losses, width, label='Perdas', color=C_LOSS, alpha=0.85)
    ax2.set_ylabel('Quantidade de Trades', fontsize=11)
    ax2.set_title('WIN RATE: Backtest vs MT5', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=11)
    ax2.legend(fontsize=10)
    
    for i, (w, l) in enumerate(zip(wins, losses)):
        total = w + l
        ax2.text(i - width/2, w + 5, f'{w/total*100:.1f}%', ha='center', fontweight='bold', fontsize=10)
        ax2.text(i + width/2, l + 5, f'{l/total*100:.1f}%', ha='center', fontweight='bold', fontsize=10)
    
    # === 3. OUTCOME DIVERGENCE ===
    ax3 = fig.add_subplot(gs[1, 1])
    both_win = (matched['bt_win'] & matched['mt5_win']).sum()
    both_lose = (~matched['bt_win'] & ~matched['mt5_win']).sum()
    bt_win_mt5_lose = (matched['bt_win'] & ~matched['mt5_win']).sum()
    bt_lose_mt5_win = (~matched['bt_win'] & matched['mt5_win']).sum()
    
    sizes = [both_win, both_lose, bt_win_mt5_lose, bt_lose_mt5_win]
    labels = ['Ambos Ganham', 'Ambos Perdem', 'BT Ganha\nMT5 Perde', 'BT Perde\nMT5 Ganha']
    colors = [C_WIN, C_LOSS, '#FFC107', '#9C27B0']
    explode = (0, 0, 0.1, 0.05)
    
    ax3.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, explode=explode, shadow=True)
    ax3.set_title('CONCORDANCIA DE RESULTADOS', fontsize=12, fontweight='bold')
    
    # === 4. SLIPPAGE ENTRADA ===
    ax4 = fig.add_subplot(gs[2, 0])
    entry_diff = matched['entry_diff']
    ax4.hist(entry_diff, bins=30, color=C_BT, alpha=0.7, edgecolor='white')
    ax4.axvline(x=0, color=C_WIN, linestyle='--', linewidth=2, label='Sem slippage')
    ax4.axvline(x=entry_diff.mean(), color=C_LOSS, linestyle='-', linewidth=2, 
                label=f'Media: {entry_diff.mean():.1f}')
    ax4.set_xlabel('Slippage (pts)', fontsize=11)
    ax4.set_ylabel('Frequencia', fontsize=11)
    ax4.set_title('SLIPPAGE NA ENTRADA (MT5 - Backtest)', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    
    # === 5. SLIPPAGE SAÍDA ===
    ax5 = fig.add_subplot(gs[2, 1])
    exit_diff = matched['exit_diff']
    ax5.hist(exit_diff, bins=30, color=C_MT5, alpha=0.7, edgecolor='white')
    ax5.axvline(x=0, color=C_WIN, linestyle='--', linewidth=2, label='Sem slippage')
    ax5.axvline(x=exit_diff.mean(), color=C_LOSS, linestyle='-', linewidth=2, 
                label=f'Media: {exit_diff.mean():.1f}')
    ax5.set_xlabel('Slippage (pts)', fontsize=11)
    ax5.set_ylabel('Frequencia', fontsize=11)
    ax5.set_title('SLIPPAGE NA SAIDA (MT5 - Backtest)', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    
    # === 6. SCATTER PLOT ===
    ax6 = fig.add_subplot(gs[3, 0])
    bt_profit_brl = matched['bt_points'] * config.point_value
    mt5_profit = matched['mt5_profit']
    colors_scatter = np.where(matched['outcome_match'], C_WIN, C_LOSS)
    
    ax6.scatter(bt_profit_brl, mt5_profit, c=colors_scatter, alpha=0.6, s=50, 
                edgecolors='white', linewidth=0.5)
    min_val = min(bt_profit_brl.min(), mt5_profit.min())
    max_val = max(bt_profit_brl.max(), mt5_profit.max())
    ax6.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    ax6.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax6.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax6.set_xlabel('Profit Backtest (R$)', fontsize=11)
    ax6.set_ylabel('Profit MT5 (R$)', fontsize=11)
    ax6.set_title('CORRELACAO DE RESULTADOS', fontsize=12, fontweight='bold')
    
    green_patch = mpatches.Patch(color=C_WIN, label='Resultado igual')
    red_patch = mpatches.Patch(color=C_LOSS, label='Resultado divergente')
    ax6.legend(handles=[green_patch, red_patch], fontsize=10)
    
    # === 7. DIVERGÊNCIA POR HORA ===
    ax7 = fig.add_subplot(gs[3, 1])
    matched['hour'] = matched['bt_timestamp'].dt.hour
    divergent_by_hour = matched[~matched['outcome_match']].groupby('hour').size()
    total_by_hour = matched.groupby('hour').size()
    divergent_rate_by_hour = (divergent_by_hour / total_by_hour * 100).fillna(0)
    
    hours = divergent_rate_by_hour.index
    ax7.bar(hours, divergent_rate_by_hour.values, color=C_LOSS, alpha=0.7, edgecolor='white')
    ax7.axhline(y=divergent_rate_by_hour.mean(), color='darkred', linestyle='--', linewidth=2,
                label=f'Media: {divergent_rate_by_hour.mean():.1f}%')
    ax7.set_xlabel('Hora do Dia', fontsize=11)
    ax7.set_ylabel('% de Trades Divergentes', fontsize=11)
    ax7.set_title('TAXA DE DIVERGENCIA POR HORA', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=10)
    
    # === 8. TABELA RESUMO ===
    ax8 = fig.add_subplot(gs[4, :])
    ax8.axis('off')
    
    summary_data = [
        ['METRICA', 'BACKTEST', 'MT5 DEMO', 'DIFERENCA'],
        ['Total de Trades', str(analysis['total_backtest_trades']), 
         str(analysis['total_mt5_trades']), '-'],
        ['Trades Matcheados', str(analysis['matched_trades']), '-', '-'],
        ['Taxa de Match', f"{analysis['match_rate_backtest']:.1f}%", '-', '-'],
        ['Win Rate', f"{analysis.get('bt_win_rate', 0):.1f}%", 
         f"{analysis.get('mt5_win_rate', 0):.1f}%", 
         f"{analysis.get('bt_win_rate', 0) - analysis.get('mt5_win_rate', 0):.1f}%"],
        ['Profit Total (R$)', f"{analysis.get('bt_total_profit_brl', 0):.2f}", 
         f"{analysis.get('mt5_total_profit', 0):.2f}",
         f"{analysis.get('bt_total_profit_brl', 0) - analysis.get('mt5_total_profit', 0):.2f}"],
        ['Slippage Entrada', '-', f"{analysis.get('avg_entry_slippage', 0):.1f} pts", '-'],
        ['Slippage Saida', '-', f"{analysis.get('avg_exit_slippage', 0):.1f} pts", '-'],
        ['Trades Divergentes', '-', 
         f"{analysis.get('divergent_trades', 0)} ({100-analysis.get('outcome_match_rate', 100):.1f}%)", '-'],
    ]
    
    table = ax8.table(cellText=summary_data, loc='center', cellLoc='center',
                      colWidths=[0.3, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Estilo do header
    for i in range(4):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax8.set_title('\nRESUMO DAS METRICAS', fontsize=12, fontweight='bold', pad=20)
    
    # Título principal
    fig.suptitle('RELATORIO DE COMPARACAO: BACKTEST vs MT5', fontsize=16, fontweight='bold', y=0.99)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Gráfico salvo: {output_path}")


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================
def compare_from_files(
    backtest_path: str,
    mt5_path: str,
    config: ComparisonConfig = None,
    output_dir: str = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Função principal para comparar arquivos de backtest e MT5.
    
    Parameters
    ----------
    backtest_path : str
        Caminho para arquivo de trades do backtest (CSV ou Parquet)
    mt5_path : str
        Caminho para arquivo de trades do MT5 (CSV ou HTML)
    config : ComparisonConfig
        Configurações de comparação
    output_dir : str, optional
        Diretório para salvar relatórios
    
    Returns
    -------
    Tuple[pd.DataFrame, dict]
        (DataFrame de trades matcheados, dicionário de análise)
    """
    if config is None:
        config = ComparisonConfig()
    
    if output_dir:
        config.output_dir = output_dir
    
    # Criar diretório de output
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Carregar dados
    print("=" * 60)
    print("CARREGANDO DADOS")
    print("=" * 60)
    bt_trades = load_backtest_trades(backtest_path)
    
    if mt5_path.endswith('.html') or mt5_path.endswith('.htm'):
        mt5_trades = load_mt5_trades_from_html(mt5_path)
    else:
        mt5_trades = load_mt5_trades_from_csv(mt5_path)
    
    # Fazer matching
    print("\n" + "=" * 60)
    print("REALIZANDO MATCHING DE TRADES")
    print("=" * 60)
    matched, unmatched_bt, unmatched_mt5 = match_trades(bt_trades, mt5_trades, config)
    
    # Analisar
    print("\n" + "=" * 60)
    print("ANALISANDO RESULTADOS")
    print("=" * 60)
    analysis = analyze_comparison(matched, unmatched_bt, unmatched_mt5, config)
    slippage = analyze_slippage(matched, config)
    analysis['slippage_analysis'] = slippage
    
    # Imprimir relatório
    report_text = print_comparison_report(matched, unmatched_bt, unmatched_mt5, analysis, config)
    
    # Salvar relatório texto
    if config.generate_txt:
        txt_path = Path(config.output_dir) / 'comparison_report.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"📄 Relatório texto salvo: {txt_path}")
    
    # Exportar Excel
    if config.generate_excel:
        excel_path = Path(config.output_dir) / 'comparison_report.xlsx'
        export_to_excel(matched, unmatched_bt, unmatched_mt5, analysis, str(excel_path), config)
    
    # Gerar gráficos
    if config.generate_plots and HAS_MATPLOTLIB:
        plot_path = Path(config.output_dir) / 'comparison_report.png'
        generate_comparison_plots(matched, analysis, str(plot_path), config)
    
    # Salvar CSV com trades matcheados
    matched_path = Path(config.output_dir) / 'matched_trades.csv'
    matched.to_csv(matched_path, index=False)
    print(f"📄 Trades matcheados salvos: {matched_path}")
    
    return matched, analysis


def compare_from_mt5_terminal(
    backtest_path: str,
    symbol: str = "WIN",
    start_date: datetime = None,
    end_date: datetime = None,
    magic_number: int = None,
    mt5_config: Dict = None,
    config: ComparisonConfig = None,
    output_dir: str = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compara backtest com dados obtidos diretamente do terminal MT5.
    
    Parameters
    ----------
    backtest_path : str
        Caminho para arquivo de trades do backtest
    symbol : str
        Símbolo para filtrar no MT5
    start_date : datetime
        Data inicial
    end_date : datetime
        Data final
    magic_number : int
        Magic number do EA
    mt5_config : dict
        Configurações de conexão MT5
    config : ComparisonConfig
        Configurações de comparação
    output_dir : str
        Diretório para salvar relatórios
    
    Returns
    -------
    Tuple[pd.DataFrame, dict]
        (DataFrame de trades matcheados, dicionário de análise)
    """
    if not HAS_MT5:
        raise ImportError("MetaTrader5 não instalado. Execute: pip install MetaTrader5")
    
    if config is None:
        config = ComparisonConfig()
    
    if output_dir:
        config.output_dir = output_dir
    
    # Criar diretório de output
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Conectar ao MT5
    print("=" * 60)
    print("CONECTANDO AO MT5")
    print("=" * 60)
    if not connect_mt5(mt5_config):
        raise RuntimeError("Falha ao conectar ao MT5")
    
    try:
        # Carregar backtest
        print("\n" + "=" * 60)
        print("CARREGANDO DADOS")
        print("=" * 60)
        bt_trades = load_backtest_trades(backtest_path)
        
        # Obter trades do MT5
        mt5_trades = get_mt5_deals(symbol, start_date, end_date, magic_number)
        
        if len(mt5_trades) == 0:
            print("Nenhum trade encontrado no MT5. Verifique os filtros.")
            return pd.DataFrame(), {}
        
        # Fazer matching
        print("\n" + "=" * 60)
        print("REALIZANDO MATCHING DE TRADES")
        print("=" * 60)
        matched, unmatched_bt, unmatched_mt5 = match_trades(bt_trades, mt5_trades, config)
        
        # Analisar
        print("\n" + "=" * 60)
        print("ANALISANDO RESULTADOS")
        print("=" * 60)
        analysis = analyze_comparison(matched, unmatched_bt, unmatched_mt5, config)
        slippage = analyze_slippage(matched, config)
        analysis['slippage_analysis'] = slippage
        
        # Imprimir relatório
        report_text = print_comparison_report(matched, unmatched_bt, unmatched_mt5, analysis, config)
        
        # Salvar relatórios
        if config.generate_txt:
            txt_path = Path(config.output_dir) / 'comparison_report.txt'
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"📄 Relatório texto salvo: {txt_path}")
        
        if config.generate_excel:
            excel_path = Path(config.output_dir) / 'comparison_report.xlsx'
            export_to_excel(matched, unmatched_bt, unmatched_mt5, analysis, str(excel_path), config)
        
        if config.generate_plots and HAS_MATPLOTLIB:
            plot_path = Path(config.output_dir) / 'comparison_report.png'
            generate_comparison_plots(matched, analysis, str(plot_path), config)
        
        matched_path = Path(config.output_dir) / 'matched_trades.csv'
        matched.to_csv(matched_path, index=False)
        print(f"📄 Trades matcheados salvos: {matched_path}")
        
        return matched, analysis
        
    finally:
        disconnect_mt5()


# =============================================================================
# CLI / MAIN
# =============================================================================
if __name__ == "__main__":
    """
    Exemplo de uso do script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Comparar Backtest vs MT5')
    parser.add_argument('--backtest', '-b', type=str, help='Caminho do arquivo de backtest')
    parser.add_argument('--mt5', '-m', type=str, help='Caminho do arquivo MT5 (ou "terminal" para conexão direta)')
    parser.add_argument('--symbol', '-s', type=str, default='WIN', help='Símbolo para filtrar (default: WIN)')
    parser.add_argument('--magic', type=int, help='Magic number do EA')
    parser.add_argument('--output', '-o', type=str, default='./reports', help='Diretório de output')
    parser.add_argument('--price-tolerance', type=float, default=30.0, help='Tolerância de preço em pontos')
    parser.add_argument('--time-tolerance', type=int, default=30, help='Tolerância de tempo em segundos')
    parser.add_argument('--point-value', type=float, default=0.20, help='Valor do ponto em R$')
    
    args = parser.parse_args()
    
    # Configuração
    config = ComparisonConfig(
        price_tolerance=args.price_tolerance,
        time_tolerance_seconds=args.time_tolerance,
        point_value=args.point_value,
        output_dir=args.output
    )
    
    if args.backtest and args.mt5:
        if args.mt5.lower() == 'terminal':
            # Conectar diretamente ao MT5
            matched, analysis = compare_from_mt5_terminal(
                backtest_path=args.backtest,
                symbol=args.symbol,
                magic_number=args.magic,
                config=config
            )
        else:
            # Comparar arquivos
            matched, analysis = compare_from_files(
                backtest_path=args.backtest,
                mt5_path=args.mt5,
                config=config
            )
    else:
        # Demo mode
        print("=" * 70)
        print("MODO DEMO - Execute com argumentos para análise real")
        print("=" * 70)
        print("\nExemplos de uso:")
        print("\n1. Comparar arquivos CSV:")
        print("   python compare_backtest_mt5_v2.py -b backtest.csv -m mt5_report.csv")
        print("\n2. Conectar diretamente ao MT5:")
        print("   python compare_backtest_mt5_v2.py -b backtest.csv -m terminal --symbol WIN --magic 12345")
        print("\n3. Com configurações customizadas:")
        print("   python compare_backtest_mt5_v2.py -b backtest.csv -m mt5.csv --price-tolerance 50 --time-tolerance 60")
        print("\nPara conexão direta ao MT5, configure as variáveis de ambiente:")
        print("   MT5_LOGIN=sua_conta")
        print("   MT5_PASSWORD=sua_senha")
        print("   MT5_SERVER=seu_servidor")
        print("   MT5_PATH=caminho_do_terminal (opcional)")