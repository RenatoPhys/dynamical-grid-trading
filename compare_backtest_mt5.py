"""
Backtest vs MetaTrader 5 Trade Comparison
==========================================
Script para comparar resultados do backtest Python com trades executados no MT5.

Author: Claude
"""

# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class ComparisonConfig:
    """Configuration for trade comparison."""
    # Tolerância para matching de preços (em pontos)
    price_tolerance: float = 5.0
    
    # Tolerância para matching de tempo (em segundos)
    time_tolerance_seconds: int = 60
    
    # Valor do ponto (para cálculo de diferenças em R$)
    point_value: float = 0.20
    
    # Timezone do MT5 (ajustar conforme necessário)
    mt5_timezone: str = 'America/Sao_Paulo'


# =============================================================================
# DATA LOADERS
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


def load_mt5_trades_from_csv(path: str) -> pd.DataFrame:
    """
    Carrega trades exportados do MT5 (formato CSV do relatório).
    
    O MT5 exporta relatórios em diferentes formatos. Esta função
    tenta detectar automaticamente o formato.
    
    Parameters
    ----------
    path : str
        Caminho para arquivo CSV exportado do MT5
    
    Returns
    -------
    pd.DataFrame
        DataFrame padronizado com trades do MT5
    """
    # Tentar diferentes encodings comuns do MT5
    encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
    
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv(path, encoding=encoding, sep=None, engine='python')
            break
        except:
            continue
    
    if df is None:
        raise ValueError(f"Não foi possível ler o arquivo: {path}")
    
    # Detectar e padronizar colunas do MT5
    df = _standardize_mt5_columns(df)
    
    print(f"MT5: {len(df)} trades carregados")
    if 'timestamp' in df.columns:
        print(f"  Período: {df['timestamp'].min()} a {df['timestamp'].max()}")
    
    return df


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


def _standardize_mt5_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza colunas do MT5 para formato comum.
    
    O MT5 pode exportar com diferentes nomes de colunas dependendo
    do idioma e versão. Esta função tenta mapear os mais comuns.
    """
    df = df.copy()
    
    # Mapeamento de possíveis nomes de colunas (PT-BR e EN)
    column_mappings = {
        'timestamp': ['Time', 'Hora', 'Open Time', 'Hora Abertura', 'Data/Hora', 
                      'Open time', 'Entry time', 'Tempo'],
        'entry_price': ['Price', 'Preço', 'Open Price', 'Preço Abertura', 
                        'Entry Price', 'Preço Entrada', 'Open price'],
        'exit_price': ['Close Price', 'Preço Fechamento', 'Exit Price', 
                       'Preço Saída', 'Close price', 'S/L', 'T/P'],
        'direction': ['Type', 'Tipo', 'Direction', 'Direção', 'Side'],
        'volume': ['Volume', 'Lots', 'Lotes', 'Size', 'Tamanho'],
        'profit': ['Profit', 'Lucro', 'P/L', 'Result', 'Resultado'],
        'symbol': ['Symbol', 'Símbolo', 'Ativo'],
        'ticket': ['Ticket', 'Order', 'Ordem', 'Deal', 'Negócio', '#'],
        'sl': ['S/L', 'Stop Loss', 'SL'],
        'tp': ['T/P', 'Take Profit', 'TP'],
        'exit_time': ['Close Time', 'Hora Fechamento', 'Exit Time', 'Close time']
    }
    
    # Aplicar mapeamento
    for standard_name, possible_names in column_mappings.items():
        for col in df.columns:
            if col in possible_names or col.lower() in [p.lower() for p in possible_names]:
                df[standard_name] = df[col]
                break
    
    # Converter timestamp
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    if 'exit_time' in df.columns:
        df['exit_time'] = pd.to_datetime(df['exit_time'], errors='coerce')
    
    # Padronizar direção
    if 'direction' in df.columns:
        df['direction'] = df['direction'].astype(str).str.upper()
        df['direction'] = df['direction'].replace({
            'BUY': 'BUY', 'COMPRA': 'BUY', 'LONG': 'BUY', 'C': 'BUY',
            'SELL': 'SELL', 'VENDA': 'SELL', 'SHORT': 'SELL', 'V': 'SELL'
        })
    
    # Converter colunas numéricas
    numeric_cols = ['entry_price', 'exit_price', 'volume', 'profit', 'sl', 'tp']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def load_mt5_from_terminal(
    symbol: str = "WIN$",
    start_date: datetime = None,
    end_date: datetime = None,
    magic_number: int = None
) -> pd.DataFrame:
    """
    Carrega trades diretamente do terminal MT5 via API Python.
    
    Requer: pip install MetaTrader5
    
    Parameters
    ----------
    symbol : str
        Símbolo para filtrar (ex: "WIN$", "WINZ24")
    start_date : datetime
        Data inicial para busca
    end_date : datetime
        Data final para busca
    magic_number : int
        Magic number do EA para filtrar
    
    Returns
    -------
    pd.DataFrame
        DataFrame com trades do MT5
    """
    try:
        import MetaTrader5 as mt5
    except ImportError:
        raise ImportError("MetaTrader5 não instalado. Execute: pip install MetaTrader5")
    
    if not mt5.initialize():
        raise RuntimeError(f"Falha ao inicializar MT5: {mt5.last_error()}")
    
    try:
        # Definir período
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        # Buscar deals (negócios executados)
        deals = mt5.history_deals_get(start_date, end_date)
        
        if deals is None or len(deals) == 0:
            print("Nenhum deal encontrado no período")
            return pd.DataFrame()
        
        # Converter para DataFrame
        df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())
        
        # Filtrar por símbolo
        if symbol:
            df = df[df['symbol'].str.contains(symbol.replace('$', ''), case=False, na=False)]
        
        # Filtrar por magic number
        if magic_number:
            df = df[df['magic'] == magic_number]
        
        # Padronizar colunas
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        df['entry_price'] = df['price']
        df['direction'] = np.where(df['type'] == 0, 'BUY', 'SELL')
        
        print(f"MT5 Terminal: {len(df)} deals carregados")
        
        return df
        
    finally:
        mt5.shutdown()


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
    
    # Adicionar índices originais
    bt['bt_idx'] = bt.index
    mt5['mt5_idx'] = mt5.index
    
    matched = []
    used_mt5_indices = set()
    
    for _, bt_row in bt.iterrows():
        best_match = None
        best_score = float('inf')
        
        for _, mt5_row in mt5.iterrows():
            if mt5_row['mt5_idx'] in used_mt5_indices:
                continue
            
            # Calcular score de matching
            score = _calculate_match_score(bt_row, mt5_row, config)
            
            if score < best_score and score < float('inf'):
                best_score = score
                best_match = mt5_row
        
        if best_match is not None:
            used_mt5_indices.add(best_match['mt5_idx'])
            matched.append({
                'bt_idx': bt_row['bt_idx'],
                'mt5_idx': best_match['mt5_idx'],
                'bt_timestamp': bt_row['timestamp'],
                'mt5_timestamp': best_match.get('timestamp'),
                'bt_direction': bt_row.get('direction'),
                'mt5_direction': best_match.get('direction'),
                'bt_entry_price': bt_row.get('entry_price'),
                'mt5_entry_price': best_match.get('entry_price'),
                'bt_exit_price': bt_row.get('exit_price'),
                'mt5_exit_price': best_match.get('exit_price'),
                'bt_points': bt_row.get('points', 0),
                'mt5_profit': best_match.get('profit', 0),
                'match_score': best_score
            })
    
    matched_df = pd.DataFrame(matched) if matched else pd.DataFrame()
    
    # Identificar não matcheados
    matched_bt_idx = set(matched_df['bt_idx']) if len(matched_df) > 0 else set()
    matched_mt5_idx = set(matched_df['mt5_idx']) if len(matched_df) > 0 else set()
    
    unmatched_bt = bt[~bt['bt_idx'].isin(matched_bt_idx)].copy()
    unmatched_mt5 = mt5[~mt5['mt5_idx'].isin(matched_mt5_idx)].copy()
    
    return matched_df, unmatched_bt, unmatched_mt5


def _calculate_match_score(bt_row: pd.Series, mt5_row: pd.Series, config: ComparisonConfig) -> float:
    """
    Calcula score de matching entre dois trades.
    Menor score = melhor match. Retorna inf se não for match válido.
    """
    score = 0.0
    
    # Verificar direção (deve ser igual)
    bt_dir = bt_row.get('direction', '').upper()
    mt5_dir = mt5_row.get('direction', '').upper()
    
    if bt_dir and mt5_dir and bt_dir != mt5_dir:
        return float('inf')
    
    # Diferença de tempo
    if 'timestamp' in bt_row and 'timestamp' in mt5_row:
        bt_time = pd.to_datetime(bt_row['timestamp'])
        mt5_time = pd.to_datetime(mt5_row['timestamp'])
        
        if pd.notna(bt_time) and pd.notna(mt5_time):
            time_diff = abs((bt_time - mt5_time).total_seconds())
            
            if time_diff > config.time_tolerance_seconds:
                return float('inf')
            
            score += time_diff / 60  # Penalidade por minuto de diferença
    
    # Diferença de preço de entrada
    bt_entry = bt_row.get('entry_price', 0)
    mt5_entry = mt5_row.get('entry_price', 0)
    
    if bt_entry and mt5_entry:
        price_diff = abs(bt_entry - mt5_entry)
        
        if price_diff > config.price_tolerance:
            return float('inf')
        
        score += price_diff
    
    return score


# =============================================================================
# ANALYSIS & REPORTING
# =============================================================================
def analyze_comparison(
    matched: pd.DataFrame,
    unmatched_bt: pd.DataFrame,
    unmatched_mt5: pd.DataFrame,
    config: ComparisonConfig = None
) -> dict:
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
        matched['entry_diff'] = matched['bt_entry_price'] - matched['mt5_entry_price']
        matched['exit_diff'] = matched['bt_exit_price'] - matched['mt5_exit_price']
        
        # Calcular diferença de tempo
        if 'bt_timestamp' in matched.columns and 'mt5_timestamp' in matched.columns:
            matched['time_diff_seconds'] = (
                pd.to_datetime(matched['bt_timestamp']) - 
                pd.to_datetime(matched['mt5_timestamp'])
            ).dt.total_seconds()
        
        results['avg_entry_diff'] = matched['entry_diff'].mean()
        results['avg_exit_diff'] = matched['exit_diff'].mean()
        results['max_entry_diff'] = matched['entry_diff'].abs().max()
        results['max_exit_diff'] = matched['exit_diff'].abs().max()
        
        if 'time_diff_seconds' in matched.columns:
            results['avg_time_diff_seconds'] = matched['time_diff_seconds'].abs().mean()
            results['max_time_diff_seconds'] = matched['time_diff_seconds'].abs().max()
        
        # Comparar resultados
        results['bt_total_points'] = matched['bt_points'].sum()
        results['mt5_total_profit'] = matched['mt5_profit'].sum()
        
        # Direções corretas
        direction_match = (matched['bt_direction'] == matched['mt5_direction']).sum()
        results['direction_match_rate'] = direction_match / len(matched) * 100
    
    return results


def print_comparison_report(
    matched: pd.DataFrame,
    unmatched_bt: pd.DataFrame,
    unmatched_mt5: pd.DataFrame,
    analysis: dict
) -> None:
    """Imprime relatório detalhado da comparação."""
    
    print("\n" + "=" * 70)
    print("RELATÓRIO DE COMPARAÇÃO: BACKTEST vs MT5")
    print("=" * 70)
    
    print("\n📊 RESUMO GERAL")
    print("-" * 40)
    print(f"Trades no Backtest:     {analysis['total_backtest_trades']}")
    print(f"Trades no MT5:          {analysis['total_mt5_trades']}")
    print(f"Trades Matcheados:      {analysis['matched_trades']}")
    print(f"Não matcheados (BT):    {analysis['unmatched_backtest']}")
    print(f"Não matcheados (MT5):   {analysis['unmatched_mt5']}")
    print(f"Taxa de Match (BT):     {analysis['match_rate_backtest']:.1f}%")
    print(f"Taxa de Match (MT5):    {analysis['match_rate_mt5']:.1f}%")
    
    if analysis['matched_trades'] > 0:
        print("\n📈 ANÁLISE DOS TRADES MATCHEADOS")
        print("-" * 40)
        print(f"Diferença média entrada:    {analysis.get('avg_entry_diff', 0):.2f} pts")
        print(f"Diferença máx entrada:      {analysis.get('max_entry_diff', 0):.2f} pts")
        print(f"Diferença média saída:      {analysis.get('avg_exit_diff', 0):.2f} pts")
        print(f"Diferença máx saída:        {analysis.get('max_exit_diff', 0):.2f} pts")
        
        if 'avg_time_diff_seconds' in analysis:
            print(f"Diferença média tempo:      {analysis['avg_time_diff_seconds']:.1f} seg")
            print(f"Diferença máx tempo:        {analysis['max_time_diff_seconds']:.1f} seg")
        
        print(f"Match de direção:           {analysis.get('direction_match_rate', 0):.1f}%")
        
        print("\n💰 COMPARAÇÃO DE RESULTADOS")
        print("-" * 40)
        print(f"Total pontos (Backtest):    {analysis.get('bt_total_points', 0):.0f}")
        print(f"Total profit (MT5):         R$ {analysis.get('mt5_total_profit', 0):.2f}")
    
    # Listar trades não matcheados
    if len(unmatched_bt) > 0:
        print("\n⚠️  TRADES DO BACKTEST NÃO ENCONTRADOS NO MT5")
        print("-" * 40)
        cols_to_show = ['timestamp', 'direction', 'entry_price', 'exit_price', 'points']
        cols_available = [c for c in cols_to_show if c in unmatched_bt.columns]
        print(unmatched_bt[cols_available].head(10).to_string(index=False))
        if len(unmatched_bt) > 10:
            print(f"... e mais {len(unmatched_bt) - 10} trades")
    
    if len(unmatched_mt5) > 0:
        print("\n⚠️  TRADES DO MT5 NÃO ENCONTRADOS NO BACKTEST")
        print("-" * 40)
        cols_to_show = ['timestamp', 'direction', 'entry_price', 'exit_price', 'profit']
        cols_available = [c for c in cols_to_show if c in unmatched_mt5.columns]
        print(unmatched_mt5[cols_available].head(10).to_string(index=False))
        if len(unmatched_mt5) > 10:
            print(f"... e mais {len(unmatched_mt5) - 10} trades")
    
    print("\n" + "=" * 70)
    
    # Diagnóstico
    print("\n🔍 DIAGNÓSTICO")
    print("-" * 40)
    
    if analysis['match_rate_backtest'] >= 95:
        print("✅ Excelente correspondência entre backtest e MT5!")
    elif analysis['match_rate_backtest'] >= 80:
        print("⚠️  Boa correspondência, mas há algumas discrepâncias.")
        print("   Verifique: slippage, latência, e horário de início.")
    elif analysis['match_rate_backtest'] >= 50:
        print("⚠️  Correspondência moderada. Possíveis causas:")
        print("   - Diferença de timezone")
        print("   - Slippage significativo")
        print("   - Configurações diferentes (SL/TP/distância)")
    else:
        print("❌ Baixa correspondência. Verifique:")
        print("   - Se os parâmetros do EA estão iguais ao backtest")
        print("   - Se o período de dados é o mesmo")
        print("   - Se há problemas de conexão/execução")


def export_comparison_report(
    matched: pd.DataFrame,
    unmatched_bt: pd.DataFrame,
    unmatched_mt5: pd.DataFrame,
    analysis: dict,
    output_path: str
) -> None:
    """
    Exporta relatório de comparação para Excel.
    
    Parameters
    ----------
    output_path : str
        Caminho para salvar o arquivo Excel
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Resumo
        summary_df = pd.DataFrame([analysis])
        summary_df.to_excel(writer, sheet_name='Resumo', index=False)
        
        # Trades matcheados
        if len(matched) > 0:
            matched.to_excel(writer, sheet_name='Matcheados', index=False)
        
        # Não matcheados
        if len(unmatched_bt) > 0:
            unmatched_bt.to_excel(writer, sheet_name='Nao_Match_Backtest', index=False)
        
        if len(unmatched_mt5) > 0:
            unmatched_mt5.to_excel(writer, sheet_name='Nao_Match_MT5', index=False)
    
    print(f"\n📁 Relatório exportado para: {output_path}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================
def compare_from_files(
    backtest_path: str,
    mt5_path: str,
    config: ComparisonConfig = None,
    export_excel: str = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Função de conveniência para comparar arquivos diretamente.
    
    Parameters
    ----------
    backtest_path : str
        Caminho para arquivo de trades do backtest (CSV ou Parquet)
    mt5_path : str
        Caminho para arquivo de trades do MT5 (CSV ou HTML)
    config : ComparisonConfig
        Configurações de comparação
    export_excel : str, optional
        Caminho para exportar relatório Excel
    
    Returns
    -------
    Tuple[pd.DataFrame, dict]
        (DataFrame de trades matcheados, dicionário de análise)
    
    Example
    -------
    >>> matched, analysis = compare_from_files(
    ...     'backtest_trades.csv',
    ...     'mt5_report.csv',
    ...     export_excel='comparison_report.xlsx'
    ... )
    """
    if config is None:
        config = ComparisonConfig()
    
    # Carregar dados
    print("Carregando dados...")
    bt_trades = load_backtest_trades(backtest_path)
    
    if mt5_path.endswith('.html') or mt5_path.endswith('.htm'):
        mt5_trades = load_mt5_trades_from_html(mt5_path)
    else:
        mt5_trades = load_mt5_trades_from_csv(mt5_path)
    
    # Fazer matching
    print("\nRealizando matching de trades...")
    matched, unmatched_bt, unmatched_mt5 = match_trades(bt_trades, mt5_trades, config)
    
    # Analisar
    analysis = analyze_comparison(matched, unmatched_bt, unmatched_mt5, config)
    
    # Imprimir relatório
    print_comparison_report(matched, unmatched_bt, unmatched_mt5, analysis)
    
    # Exportar se solicitado
    if export_excel:
        export_comparison_report(matched, unmatched_bt, unmatched_mt5, analysis, export_excel)
    
    return matched, analysis


def quick_validate(
    backtest_trades: pd.DataFrame,
    mt5_trades: pd.DataFrame
) -> bool:
    """
    Validação rápida: verifica se número de trades e resultado geral são similares.
    
    Returns
    -------
    bool
        True se validação passou, False caso contrário
    """
    bt_count = len(backtest_trades)
    mt5_count = len(mt5_trades)
    
    count_diff = abs(bt_count - mt5_count) / max(bt_count, mt5_count, 1)
    
    print(f"Trades Backtest: {bt_count}")
    print(f"Trades MT5: {mt5_count}")
    print(f"Diferença: {count_diff*100:.1f}%")
    
    if count_diff > 0.1:  # Mais de 10% de diferença
        print("⚠️  Diferença significativa no número de trades!")
        return False
    
    print("✅ Número de trades similar")
    return True


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    """
    Exemplo de uso do script.
    
    Para usar, ajuste os caminhos dos arquivos abaixo:
    """
    
    # =========================
    # CONFIGURAÇÃO
    # =========================
    config = ComparisonConfig(
        price_tolerance=10.0,        # Tolerância de 10 pontos para preços
        time_tolerance_seconds=120,  # Tolerância de 2 minutos
        point_value=0.20             # Valor do ponto WIN
    )
    
    # =========================
    # OPÇÃO 1: Comparar arquivos CSV/Parquet
    # =========================
    """
    # Descomente e ajuste os caminhos:
    
    matched, analysis = compare_from_files(
        backtest_path='backtest_trades.csv',       # Exportar do seu backtest
        mt5_path='mt5_statement.csv',              # Exportar do MT5: File > Save as Report
        config=config,
        export_excel='comparison_report.xlsx'
    )
    """
    
    # =========================
    # OPÇÃO 2: Carregar diretamente do MT5 Terminal
    # =========================
    """
    # Descomente para usar API do MT5:
    
    from datetime import datetime
    
    mt5_trades = load_mt5_from_terminal(
        symbol="WIN",
        start_date=datetime(2024, 2, 1),
        end_date=datetime(2024, 2, 28),
        magic_number=123456
    )
    
    bt_trades = load_backtest_trades('backtest_trades.csv')
    
    matched, unmatched_bt, unmatched_mt5 = match_trades(bt_trades, mt5_trades, config)
    analysis = analyze_comparison(matched, unmatched_bt, unmatched_mt5, config)
    print_comparison_report(matched, unmatched_bt, unmatched_mt5, analysis)
    """
    
    # =========================
    # DEMO: Criar dados de exemplo
    # =========================
    print("=" * 70)
    print("DEMO: Executando com dados de exemplo")
    print("=" * 70)
    
    # Simular dados de backtest
    demo_bt = pd.DataFrame({
        'timestamp': pd.date_range('2024-02-02 09:00', periods=5, freq='15min'),
        'direction': ['BUY', 'SELL', 'BUY', 'SELL', 'BUY'],
        'entry_price': [127500, 127600, 127400, 127550, 127450],
        'exit_price': [127520, 127570, 127450, 127500, 127480],
        'points': [20, 30, 50, 50, 30],
        'status': [1, 1, 1, 1, 1]
    })
    
    # Simular dados do MT5 (com pequenas diferenças)
    demo_mt5 = pd.DataFrame({
        'timestamp': pd.date_range('2024-02-02 09:00', periods=5, freq='15min') + pd.Timedelta(seconds=5),
        'direction': ['BUY', 'SELL', 'BUY', 'SELL', 'BUY'],
        'entry_price': [127502, 127598, 127405, 127548, 127452],
        'exit_price': [127522, 127568, 127455, 127498, 127482],
        'profit': [4.0, 6.0, 10.0, 10.0, 6.0]
    })
    
    # Executar comparação
    matched, unmatched_bt, unmatched_mt5 = match_trades(demo_bt, demo_mt5, config)
    analysis = analyze_comparison(matched, unmatched_bt, unmatched_mt5, config)
    print_comparison_report(matched, unmatched_bt, unmatched_mt5, analysis)
    
    print("\n" + "=" * 70)
    print("Para usar com seus dados reais, ajuste os caminhos no código acima!")
    print("=" * 70)
