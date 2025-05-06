import numpy as np
import pandas as pd
from tensorflow.keras.layers import LSTM, Dense
from fredapi import Fred
import argparse
import os
import json
import logging
from xgboost import XGBRegressor
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
import joblib
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras_tuner import Hyperband
from keras_tuner.engine.hyperparameters import HyperParameters
from tensorflow.keras.layers import Layer
import random
from copy import deepcopy
from pathlib import Path
from tensorflow.keras.saving import register_keras_serializable

print("✅ TensorFlow version:", tf.__version__)
print("🖥️  GPU disponibili:", tf.config.list_physical_devices('GPU'))
print("⚙️  CUDA attivo:", tf.test.is_built_with_cuda())

# Semina tutto
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Mappa dei valori pip per varie coppie forex
pip_values = {
    'EURUSD': 0.0001, 'USDJPY': 0.01, 'GBPUSD': 0.0001, 'USDCHF': 0.0001,
    'USDCAD': 0.0001, 'AUDUSD': 0.0001, 'NZDUSD': 0.0001,
    'EURGBP': 0.0001, 'EURJPY': 0.01, 'GBPJPY': 0.01, 'CHFJPY': 0.01,
    'EURCHF': 0.0001, 'EURAUD': 0.0001, 'EURNZD': 0.0001, 'EURCAD': 0.0001,
    'GBPCAD': 0.0001, 'GBPCHF': 0.0001, 'GBPAUD': 0.0001, 'GBPNZD': 0.0001,
    'AUDJPY': 0.01, 'AUDNZD': 0.0001, 'AUDCAD': 0.0001, 'AUDCHF': 0.0001,
    'CADJPY': 0.01, 'CADCHF': 0.0001, 'NZDJPY': 0.01, 'NZDCHF': 0.0001,
    'NZDCAD': 0.0001,
    'EURSEK': 0.0001, 'EURNOK': 0.0001, 'EURDKK': 0.0001, 'EURTRY': 0.0001,
    'USDHKD': 0.0001, 'USDSGD': 0.0001, 'USDZAR': 0.0001, 'USDMXN': 0.0001,
    'USDNOK': 0.0001, 'USDSEK': 0.0001, 'USDTRY': 0.0001, 'USDPLN': 0.0001,
    'USDCNH': 0.0001,
    'XAUUSD': 0.01, 'XAGUSD': 0.01
}

logging.basicConfig(
    filename='logs.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def print_log(msg):
    """Funzione per registrare nei log e stampare a console"""
    logging.info(msg)
    print(msg)

# ------------------ LOSS FUNZIONE: QUANTILE ------------------
@register_keras_serializable(package="Custom")
def quantile_loss(q):
    def loss(y_true, y_pred):
        e = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * e, (q - 1) * e))
    return loss

# ------------------ ATTENTION LAYER ------------------
@register_keras_serializable(package="Custom")
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], 1), initializer='normal')
        self.b = self.add_weight(name='att_bias', shape=(input_shape[1], 1), initializer='zeros')
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    for i in range(seq_length, len(X)):
        Xs.append(X[i - seq_length:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def select_features_with_model(df, feature_columns, targets, top_n=50):
    """
    Seleziona le feature più rilevanti usando un XGBoostRegressor addestrato su entrambi i target.
    """
    X = df[feature_columns]
    y = df[targets].mean(axis=1)  # media tra High e Low come proxy target

    model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    model.fit(X, y)

    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': importances
    }).sort_values(by="importance", ascending=False)

    selected_features = feature_importance_df.head(top_n)["feature"].tolist()

    print_log(f"✅ Feature selezionate dal modello (top {top_n}): {selected_features}")
    return selected_features



def calculate_pivot_points(df):
    """
    Calcola i Pivot Point giornalieri, settimanali e mensili e li aggiunge al DataFrame.
    """
    df = df.copy()
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)

    for prefix in ['D_', 'W_', 'M_']:
        cols_to_drop = [col for col in df.columns if col.startswith(prefix)]
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Calcolo dei Pivot Point giornalieri
    daily = (
        df.resample("D")
        .agg({"High": "max", "Low": "min", "Close": "last"})
        .shift(1)                 # 👈  giorno precedente
    )
    daily['PP'] = (daily['High'] + daily['Low'] + daily['Close']) / 3
    daily['R1'] = (2 * daily['PP']) - daily['Low']
    daily['S1'] = (2 * daily['PP']) - daily['High']
    daily['R2'] = daily['PP'] + (daily['High'] - daily['Low'])
    daily['S2'] = daily['PP'] - (daily['High'] - daily['Low'])
    daily = daily[['PP', 'R1', 'S1', 'R2', 'S2']]
    daily.columns = ['D_PP', 'D_R1', 'D_S1', 'D_R2', 'D_S2']
    df = df.merge(daily, left_index=True, right_index=True, how='left')

    # Calcolo dei Pivot Point settimanali
    weekly = (
        df.resample("W-MON")        # settimana che termina la domenica
        .agg({"High": "max", "Low": "min", "Close": "last"})
        .shift(1)                 # 👈  settimana precedente
    )
    weekly['PP'] = (weekly['High'] + weekly['Low'] + weekly['Close']) / 3
    weekly['R1'] = (2 * weekly['PP']) - weekly['Low']
    weekly['S1'] = (2 * weekly['PP']) - weekly['High']
    weekly['R2'] = weekly['PP'] + (weekly['High'] - weekly['Low'])
    weekly['S2'] = weekly['PP'] - (weekly['High'] - weekly['Low'])
    weekly = weekly[['PP', 'R1', 'S1', 'R2', 'S2']]
    weekly.columns = ['W_PP', 'W_R1', 'W_S1', 'W_R2', 'W_S2']
    df = df.merge(weekly, left_index=True, right_index=True, how='left')

    # Calcolo dei Pivot Point mensili
    monthly = (
        df.resample("ME")           # ultimo giorno del mese
        .agg({"High": "max", "Low": "min", "Close": "last"})
        .shift(1)                 # 👈  mese precedente
    )
    monthly['PP'] = (monthly['High'] + monthly['Low'] + monthly['Close']) / 3
    monthly['R1'] = (2 * monthly['PP']) - monthly['Low']
    monthly['S1'] = (2 * monthly['PP']) - monthly['High']
    monthly['R2'] = monthly['PP'] + (monthly['High'] - monthly['Low'])
    monthly['S2'] = monthly['PP'] - (monthly['High'] - monthly['Low'])
    monthly = monthly[['PP', 'R1', 'S1', 'R2', 'S2']]
    monthly.columns = ['M_PP', 'M_R1', 'M_S1', 'M_R2', 'M_S2']
    df = df.merge(monthly, left_index=True, right_index=True, how='left')

    df.reset_index(inplace=True)
    return df

def add_slim_features(df):
    """
    Aggiunge un set ridotto di feature, inclusi pivot settimanali e mensili.
    Ritorna lo stesso DataFrame con colonne aggiuntive:
      - RSI
      - MACD
      - Bollinger (upper/lower)
      - OBV
      - Tenkan_sen, Kijun_sen (Ichimoku parziale)
      - Pattern Candlestick (Bullish_Engulfing, Bearish_Engulfing, Hammer, Shooting_Star)
      - DayOfWeek, DayOfMonth
      - Pivot Points Weekly/Monthly: Pivot, R1, S1, R2, S2
    """

    # ===== 1) RSI =====
    def calculate_RSI(series, window=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))
    
    # Rimozione sicura delle colonne se già presenti
    cols_to_drop = [
        "DayOfWeek", "DayOfWeek_sin", "DayOfWeek_cos",
        "DayOfMonth", "Hour", "Hour_sin", "Hour_cos",
        "Body_Size", "Is_Bullish", "Is_Bearish",
        "Trend_Strength", "Gap_Up", "Gap_Down"
    ]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    
    df['RSI'] = calculate_RSI(df['Close'], window=14)
    
    # ===== 2) MACD (senza linea di segnale) =====
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = (ema12 - ema26) * 100
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # ===== 3) Bollinger Bands (upper/lower) =====
    rolling_mean20 = df['Close'].rolling(window=20).mean()
    rolling_std20  = df['Close'].rolling(window=20).std()
    df['Bollinger_upper'] = rolling_mean20 + (2 * rolling_std20)
    df['Bollinger_lower'] = rolling_mean20 - (2 * rolling_std20)

    # ===== 4) OBV (On Balance Volume) =====
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # ===== 5) Ichimoku (Tenkan_sen e Kijun_sen) =====
    #  (Non aggiungiamo Senkou_Span e Chikou_Span per ridurre correlazioni)
    period1 = 9
    period2 = 26
    df['Tenkan_sen'] = (
        df['High'].rolling(window=period1).max() +
        df['Low'].rolling(window=period1).min()
    ) / 2
    df['Kijun_sen'] = (
        df['High'].rolling(window=period2).max() +
        df['Low'].rolling(window=period2).min()
    ) / 2

    # ===== 6) Pattern Candlestick (discreti) =====
    df['Bullish_Engulfing'] = 0
    df['Bearish_Engulfing'] = 0
    df['Hammer'] = 0
    df['Shooting_Star'] = 0

    for i in range(1, len(df)):
        prev_open  = df.at[i-1, 'Open']
        prev_close = df.at[i-1, 'Close']
        curr_open  = df.at[i, 'Open']
        curr_close = df.at[i, 'Close']

        if (prev_close < prev_open) and (curr_close > curr_open) \
           and (curr_close > prev_open) and (curr_open < prev_close):
            df.at[i, 'Bullish_Engulfing'] = 1
        
        if (prev_close > prev_open) and (curr_close < curr_open) \
           and (curr_close < prev_open) and (curr_open > prev_close):
            df.at[i, 'Bearish_Engulfing'] = 1

        body = abs(curr_close - curr_open)
        day_range = df.at[i, 'High'] - df.at[i, 'Low']
        if day_range > 0:
            if curr_close > curr_open:
                lower_wick = (curr_open - df.at[i, 'Low'])
                upper_wick = (df.at[i, 'High'] - curr_close)
            else:
                lower_wick = (curr_close - df.at[i, 'Low'])
                upper_wick = (df.at[i, 'High'] - curr_open)

            # Hammer
            if (body > 0) and (lower_wick >= 2 * body) and (upper_wick <= 0.5 * body):
                df.at[i, 'Hammer'] = 1

            # Shooting_Star
            if (body > 0) and (upper_wick >= 2 * body) and (lower_wick <= 0.5 * body):
                df.at[i, 'Shooting_Star'] = 1

    # === Feature temporali migliorate per D1 ===
    # Giorno della settimana (0 = lunedì, 6 = domenica)
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek

    # Codifica ciclica del giorno della settimana (per LSTM o modelli non lineari)
    df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
    df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

    # Giorno del mese (utile per pattern economici o mensili)
    df['DayOfMonth'] = df['DateTime'].dt.day

    df['DateTime_norm'] = pd.to_datetime(df['DateTime']).dt.normalize()
    df.sort_values('DateTime_norm', inplace=True)

    df = calculate_pivot_points(df)
    return df

# Inserisci qui la tua API Key FRED
FRED_API_KEY = "67866bca2e902f26ae62a679f15ec1f2"

# Dizionario delle serie FRED (macro) per le varie valute

macros = {
    'USD': {'GDP': 'GDP', 'Inflation': 'CPIAUCSL', 'Unemployment': 'UNRATE', 'InterestRate': 'FEDFUNDS'},
    'EUR': {'GDP': 'NAEXKP01EZQ657S', 'Inflation': 'CP0000EZ19M086NEST', 'Unemployment': 'LRHUTTTTEZM156S', 'InterestRate': 'IRSTCI01EZM156N'},
    'GBP': {'GDP': 'CLVMNACSCAB1GQUK', 'Inflation': 'GBRCPIALLMINMEI', 'Unemployment': 'LRHUTTTTGBM156S', 'InterestRate': 'IR3TIB01GBM156N'},
    'JPY': {'GDP': 'NAEXKP01JPQ657S', 'Inflation': 'JPNCPIALLMINMEI', 'Unemployment': 'LRHUTTTTJPQ156S', 'InterestRate': 'IR3TIB01JPM156N'},
    'AUD': {'GDP': 'NAEXKP01AUQ661S', 'Inflation': 'AUSCPIALLQINMEI', 'Unemployment': 'LRHUTTTTAUM156S', 'InterestRate': 'IR3TIB01AUM156N'},
    'CAD': {'GDP': 'NAEXKP01CAQ657S', 'Inflation': 'CANCPIALLMINMEI', 'Unemployment': 'LRHUTTTTCAQ156S', 'InterestRate': 'IR3TIB01CAM156N'},
    'CHF': {'GDP': 'CLVMNACSCAB1GQCH', 'Inflation': 'CPALTT01CHQ659N', 'Unemployment': 'LRHUTTTTCHQ156S', 'InterestRate': 'IR3TIB01CHM156N'},
    'NZD': {'GDP': 'NAEXKP01NZQ661S', 'Inflation': 'NZLCPIALLQINMEI', 'Unemployment': 'LRHUTTTTNZQ156S', 'InterestRate': 'IR3TIB01NZM156N'}
}

###############################################################################
# Funzione per recuperare i codici base e quote da una coppia, es. "AUDNZD" -> ("AUD","NZD")
###############################################################################
def split_symbol_into_currencies(symbol):
    """
    Tenta di estrarre base e quote prendendo i primi 3 char come base e i successivi 3 come quote.
    Esempio: 'AUDNZD' -> ('AUD','NZD')
    """
    base = symbol[:3].upper()
    quote = symbol[3:].upper()
    return base, quote

def fetch_oil_data(start_date, end_date):
    fred = Fred(api_key=FRED_API_KEY)
    oil_series = fred.get_series('DCOILWTICO', observation_start=start_date, observation_end=end_date)
    df_oil = oil_series.to_frame(name="Oil_WTI").sort_index()
    df_oil.index.name = 'DateTime'
    df_oil.ffill(inplace=True)
    return df_oil

###############################################################################
# Funzione per fare fetch dei dati macro da FRED e restituirli in frequenza giornaliera
###############################################################################
def fetch_macro_data_for_currency(currency, start_date, end_date):
    """
    Recupera da FRED le serie economiche indicate in `macros[currency]` 
    e restituisce un DataFrame in frequenza giornaliera con forward-fill.
    """
    if currency not in macros:
        # Se non c'è nel dizionario macros, ritorniamo df vuoto
        return pd.DataFrame()

    fred = Fred(api_key=FRED_API_KEY)
    series_map = macros[currency]

    # Creiamo un DF vuoto con un index giornaliero
    idx = pd.date_range(start_date, end_date, freq='D')
    df_macro = pd.DataFrame(index=idx)

    # Per ogni indicatore, fetch da FRED e mergiamo nella df
    for macro_key, fred_series_id in series_map.items():
        series_data = fred.get_series(fred_series_id, observation_start=start_date, observation_end=end_date)
        # series_data è una pd.Series con index = date, value = valore dell'indicatore
        # Rinominiamo la colonna in currency_macro_key (es. 'AUD_GDP')
        temp_df = series_data.to_frame(name=f"{currency}_{macro_key}").sort_index()
        # Merge / Join su index
        df_macro = df_macro.join(temp_df, how='left')

    # Forward fill per riempire i buchi (es. da mensile a giornaliero)
    df_macro.ffill(inplace=True)
    df_macro.bfill(inplace=True)

    return df_macro

###############################################################################
# Funzione che integra i dati macro di base e quote in un singolo DF
###############################################################################
def integrate_macro_data(df, symbol):
    """
    Integra i dati macroeconomici FRED su base giornaliera all'interno di un dataframe D1,
    usando merge_asof per unire in modo temporale coerente.
    """
    if len(df) == 0:
        return df

    macro_prefixes = ["Diff_", "Oil_", "USD_", "EUR_", "AUD_", "NZD_", "CAD_", "JPY_", "CHF_", "GBP_"]
    for prefix in macro_prefixes:
        cols_to_drop = [col for col in df.columns if col.startswith(prefix)]
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Parsing corretto del DateTime D1
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df = df.sort_values('DateTime')

    base_cur, quote_cur = split_symbol_into_currencies(symbol)
    start_date = df['DateTime'].min().date().isoformat()
    end_date   = df['DateTime'].max().date().isoformat()

    # Fetch dati macro separati
    df_base = fetch_macro_data_for_currency(base_cur, start_date, end_date)
    df_quote = fetch_macro_data_for_currency(quote_cur, start_date, end_date)

    # Merge concatenato tra base e quote
    df_macro = pd.concat([df_base, df_quote], axis=1)

    # Calcolo differenze macro (es. USD_GDP - EUR_GDP)
    common_macros = set(df_base.columns).intersection(set(df_quote.columns))
    for macro_col in common_macros:
        base_val = df_base[macro_col]
        quote_val = df_quote[macro_col]
        macro_name = macro_col.split('_')[-1]  # es. 'InterestRate'
        df_macro[f"Diff_{macro_name}"] = base_val.values - quote_val.values

    # Reset e rinomina index per merge_asof
    df_macro.reset_index(inplace=True)  # index è la data
    df_macro.rename(columns={'index': 'DateTime'}, inplace=True)
    df_macro['DateTime'] = pd.to_datetime(df_macro['DateTime'])

    # Oil (se CAD coinvolto)
    if 'CAD' in (base_cur, quote_cur):
        df_oil = fetch_oil_data(start_date, end_date)
        df_oil.reset_index(inplace=True)
        df_oil['DateTime'] = pd.to_datetime(df_oil['DateTime'])
        df_macro = pd.merge_asof(df_macro.sort_values('DateTime'), df_oil.sort_values('DateTime'), on='DateTime', direction='backward')

    # Merge temporale coerente: macro daily -> candele D1
    df = pd.merge_asof(
        df.sort_values('DateTime'),
        df_macro.sort_values('DateTime'),
        on='DateTime',
        direction='backward'
    )

    # Fill dei buchi eventuali
    df.ffill(inplace=True)

    return df

# Aggiungi questa funzione al tuo codice:
def check_and_clean_dataframe(df, feature_cols, target_cols):
    """
    Verifica e pulisce NaN e Inf in un DataFrame in modo sicuro.
    Ignora colonne non numeriche per controlli np.isinf/np.isnan.
    """
    print_log("\n🔍 Verifica dati per problemi:")

    for col in feature_cols:
        col_dtype = df[col].dtype

        # Salta colonne non numeriche
        if not np.issubdtype(col_dtype, np.number):
            continue

        nan_count = df[col].isna().sum()
        inf_count = np.isinf(df[col]).sum()

        if nan_count > 0:
            print_log(f"⚠️ Trovati {nan_count} valori NaN in {col}")
            if col in ['RSI', 'MACD', 'MACD_signal', 'Bollinger_upper', 'Bollinger_lower',
                       'Tenkan_sen', 'Kijun_sen', 'OBV']:
                df[col] = df[col].ffill().bfill()
            else:
                df[col] = df[col].fillna(df[col].median())

        if inf_count > 0:
            print_log(f"⚠️ Trovati {inf_count} valori Inf in {col}")
            df[col] = df[col].replace([np.inf, -np.inf], df[col].median())

        # Outlier estremi
        mean = df[col].mean()
        std = df[col].std()
        outliers = ((df[col] > mean + 10*std) | (df[col] < mean - 10*std)).sum()
        if outliers > 0:
            print_log(f"⚠️ Trovati {outliers} outlier estremi in {col}")
            # Optional: puoi clipparli
            # df[col] = df[col].clip(mean - 5*std, mean + 5*std)

    # Controllo finale sui target
    for col in target_cols:
        if col not in df.columns:
            continue
        if df[col].isna().sum() > 0 or np.isinf(df[col]).sum() > 0:
            print_log(f"❌ Valori problematici nella colonna target {col}!")
            df[col] = df[col].ffill().bfill()

    return df

def add_directional_features(df):
    df = df.copy()

    # Feature originali
    df['Body_Size'] = ((df['Close'] - df['Open']).abs()) * 100
    df['Is_Bullish'] = (df['Close'] > df['Open']).astype(int)
    df['Is_Bearish'] = (df['Close'] < df['Open']).astype(int)

    # Range & rapporto corpo/range
    df['Range'] = df['High'] - df['Low']
    df['Trend_Strength'] = df['Body_Size'] / (df['Range'] + 1e-6)

    # Gap e variazione rispetto a ieri
    df['Previous_Close'] = df['Close'].shift(1)
    df['Previous_Open'] = df['Open'].shift(1)
    df['Gap_Open'] = df['Open'] - df['Previous_Close']
    df['Close_vs_Previous_Close'] = (df['Close'] - df['Previous_Close']) / df['Previous_Close']
    
    # NUOVE FEATURE DIREZIONALI
    # Rapporto Up/Down per catturare la forza direzionale
    df['Up_Down_Ratio'] = (df['Body_Size'] * df['Is_Bullish']) / (df['Body_Size'] * df['Is_Bearish'] + 1e-9)
    
    # Tendenza di lungo periodo (ultimi 10 periodi)
    df['Trend_Direction_10'] = df['Close'].diff(10).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    
    # Tendenza a breve termine (ultimi 5 periodi)
    df['Trend_Direction_5'] = df['Close'].diff(5).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    
    # Price Momentum (accelerazione dei prezzi)
    df['Price_Momentum'] = df['Close'].diff().diff()
    
    # Rapporto High-Open rispetto a Open-Low (asimmetria del range)
    df['Range_Asymmetry'] = (df['High'] - df['Open']) / (df['Open'] - df['Low'] + 1e-9)

    df['High_vs_PrevHigh'] = df['High'] - df['High'].shift(1)
    df['Low_vs_PrevLow'] = df['Low'] - df['Low'].shift(1)
    df['Volatility_Ratio'] = df['Range'] / (df['Range'].rolling(window=5).mean() + 1e-9)
    df['Volume_Spike'] = df['Volume'] / (df['Volume'].rolling(window=10).mean() + 1e-9)
    rolling_mean20 = df['Close'].rolling(window=20).mean()
    rolling_std20 = df['Close'].rolling(window=20).std()
    df['BB_Width'] = (2 * rolling_std20) / (rolling_mean20 + 1e-9)

    # === Classificazione del comportamento della barra ===
    high_delta = df['High'] - df['Open']
    low_delta = df['Open'] - df['Low']
    max_delta = df[['High', 'Low']].sub(df['Open'], axis=0).abs().max(axis=1)

    relative_diff = (high_delta - low_delta) / (max_delta + 1e-9)

    conditions = [
        (relative_diff > 0.4),     # breakout verso l’alto
        (relative_diff < -0.4),    # breakout verso il basso
    ]
    choices = ['breakout_up', 'breakout_down']
    df['Bar_Type'] = np.select(conditions, choices, default='neutral')
    bar_type_map = {'neutral': 0, 'breakout_up': 1, 'breakout_down': 2}
    df['Bar_Type_Code'] = df['Bar_Type'].map(bar_type_map)
    
    # Drop helper temporanei
    df.drop(columns=['Previous_Close', 'Previous_Open'], inplace=True)

    return df

def load_and_update_data(symbol): 
    df_main = pd.read_csv(f"merged\{symbol}1440.csv", sep=",")
    df_new = pd.read_csv(f"downloaded\D1_data_{symbol}.csv", sep="\t", header=None)
    
    df_new.columns = ["DateTime", "Open", "High", "Low", "Close", "Volume"]

    df_main.sort_values('DateTime', inplace=True)
    df_main.reset_index(drop=True, inplace=True)
    df_new.sort_values('DateTime', inplace=True)
    df_new.reset_index(drop=True, inplace=True)

    
    if df_new.iloc[-1]['Volume'] < 3000:
        df_new = df_new.iloc[:-1]
        print_log("Rimossa ultima riga con volume insufficiente")

    df_main["DateTime"] = pd.to_datetime(df_main["DateTime"], errors='coerce')
    df_new["DateTime"] = pd.to_datetime(df_new["DateTime"], errors='coerce')

    last_time = df_main['DateTime'].max()
    df_new_filtered = df_new[df_new['DateTime'] > last_time]

    df_new_filtered = df_new_filtered[[col for col in df_main.columns if col in df_new_filtered.columns]]

    df_updated = pd.concat([df_main, df_new_filtered], ignore_index=True)

    df_updated["DateTime"] = pd.to_datetime(df_updated["DateTime"], errors='coerce')

    df_updated = add_slim_features(df_updated)
    df_updated = add_directional_features(df_updated)
    df_updated = integrate_macro_data(df_updated, symbol)

    # Applica la formattazione numerica ai prezzi
    for col in price_cols:
        if col in df_updated.columns:
            df_updated[col] = df_updated[col].round(decimal_precision)

    df_updated.to_csv(f"merged\{symbol}1440.csv", index=False, sep=",", header=True)

    return df_updated

def save_selected_features(selected_features, data_folder, symbol):
    """Salva le features selezionate in un file JSON"""
    features_path = os.path.join(data_folder, f"models/features_{symbol}.json")
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    with open(features_path, 'w') as f:
        json.dump(selected_features, f)
    print_log(f"✅ Features selezionate salvate in: {features_path}")

def load_selected_features(data_folder, symbol):
    """Carica le features selezionate da un file JSON"""
    features_path = os.path.join(data_folder, f"models/features_{symbol}.json")
    if os.path.exists(features_path):
        with open(features_path, 'r') as f:
            return json.load(f)
    return None

def asymmetric_mse_high(y_true, y_pred):
    """Loss personalizzata per l'output High che penalizza di più le sottostime"""
    error = y_true - y_pred
    # Penalizza sottostime (error > 0) più pesantemente delle sovrastime (error < 0)
    return tf.reduce_mean(tf.where(error > 0, 2.0 * tf.square(error), 0.5 * tf.square(error)))

def asymmetric_mse_low(y_true, y_pred):
    """Loss personalizzata per l'output Low che penalizza di più le sovrastime"""
    error = y_true - y_pred
    # Penalizza sovrastime (error < 0) più pesantemente delle sottostime (error > 0)
    return tf.reduce_mean(tf.where(error < 0, 2.0 * tf.square(error), 0.5 * tf.square(error)))

def load_best_model_and_hp(symbol: str, data_folder: str, seq_len: int):
    models_dir = Path(data_folder) / "models"
    model_path = models_dir / f"model_{symbol}.keras"
    scaler_feat_path = models_dir / f"scaler_features_{symbol}.joblib"
    scaler_y_path    = models_dir / f"scaler_y_{symbol}.joblib"
    hp_path          = models_dir / f"best_hp_{symbol}.json"

    # ––– carica artefatti
    custom_objects = {
        "Attention": Attention,
        "quantile_loss": quantile_loss,
        "asymmetric_mse_high": asymmetric_mse_high,
        "asymmetric_mse_low": asymmetric_mse_low,
    }
    model = tf.keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)
    scaler_features = joblib.load(scaler_feat_path)
    scalers = joblib.load(scaler_y_path)
    scaler_y_high = scalers["high"]
    scaler_y_low  = scalers["low"]
    scaler_y_body = scalers["body"]

    with open(hp_path) as fp:
        hp_values = json.load(fp)

    # ➜ fallback se "num_features" manca
    num_features = hp_values.get("num_features", model.input_shape[-1])
    hp_values["num_features"] = num_features      # opzionale: rimettilo in memoria

    return model, scaler_features, scaler_y_high, scaler_y_low, scaler_y_body, hp_values

# ------------------ MODELLO CON ATTENTION & QUANTILE ------------------
def build_model(hp, num_features):
    input_layer = Input(shape=(SEQ_LEN, num_features))

    shared = LSTM(
        hp.Choice("shared_units", [32, 64, 128]),
        return_sequences=True,
        kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)
    )(input_layer)
    shared = Dropout(hp.Float("shared_dropout", 0.1, 0.5, step=0.1))(shared)

    attention = Attention()(shared)

    high_branch = Dense(hp.Int("high_dense", 4, 16, step=4), activation="relu")(attention)
    high_output = Dense(1, name='high_output')(high_branch)

    low_branch = Dense(hp.Int("low_dense", 4, 16, step=4), activation="relu")(attention)
    low_output  = Dense(1, name='low_output')(low_branch)

    # --- branch terzo output (corpo) ------------------------------
    body_branch = Dense(hp.Int("body_dense", 4, 16, step=4), activation="relu")(attention)
    body_output = Dense(1, name='body_output', activation="tanh")(body_branch)   # −1 … +1

    model = Model(inputs=input_layer,
                outputs=[high_output, low_output, body_output])

    optimizer = Adam(learning_rate=hp.Float("lr", 1e-4, 1e-2, sampling="log"))
    model.compile(
        optimizer=optimizer,
        loss={
            'high_output': quantile_loss(0.80),
            'low_output':  quantile_loss(0.20),
            'body_output': 'mse'          
        },
        loss_weights={'high_output': 1.0, 'low_output': 1.0, 'body_output': 0.5}
    )
    return model

parser = argparse.ArgumentParser()
parser.add_argument("--symbol", required=True, help="Simbolo della coppia (es. EURUSD)")
parser.add_argument("--data_folder", required=True, help="Cartella dove salvare il modello")
parser.add_argument("--next_open", type=float, help="Open della prossima barra (opzionale)")
parser.add_argument(
    "--force_retrain",
    action="store_true",
    help="Forza l'addestramento del modello anche se già esiste"
)
args = parser.parse_args()

symbol = args.symbol.upper()
# Identifica se stiamo lavorando con una coppia JPY
is_jpy_pair = "JPY" in symbol.upper()
jpy_scale_factor = 100.0 if is_jpy_pair else 1.0
model_path = os.path.join(args.data_folder, f"models\model_{symbol}.keras")

decimal_precision = 3 if "JPY" in symbol.upper() else 5
price_cols = ["Open", "High", "Low", "Close"]

df = load_and_update_data(symbol)
targets = ["High", "Low"]

# Prima della pulizia
print_log(f"Ultimi 5 valori OHLC prima della pulizia: {df[['DateTime', 'Open', 'High', 'Low', 'Close']].tail(5)}")

df = check_and_clean_dataframe(df, df.columns.difference(['DateTime']), targets)

# Dopo la pulizia
print_log(f"Ultimi 5 valori OHLC dopo la pulizia: {df[['DateTime', 'Open', 'High', 'Low', 'Close']].tail(5)}")

all_numeric = df.select_dtypes(include=[np.number])

features_path = os.path.join(args.data_folder, f"models/features_{symbol}.json")
selected = None
model_exists = os.path.exists(model_path)
# Verifica se dobbiamo riselezionare le features
if args.force_retrain or not model_exists or not os.path.exists(features_path):
    print_log(f"\n🔍 Selezione features per {symbol}...")

    # Step 1: Feature base obbligatorie
    base_features = ["Open", "Close", "Volume"]

    # Step 2: Feature ritenute ridondanti/duplicate
    redundant_features = set([
        "DayOfWeek",
        "Hour",
        "M_PP", "M_R1", "M_R2", "M_S1", "M_S2",
        "Bullish_Engulfing", "Bearish_Engulfing", "Hammer", "Shooting_Star",
        "Gap_Open",
        "Trend_Strength",
        "MACD_signal",
        "OBV"
    ])

    # Step 3: Costruzione candidate features
    extra_features = [
        col for col in all_numeric.columns
        if col not in base_features + targets + ["DateTime"]
        and col not in redundant_features
    ]

    candidates = base_features + extra_features

    # Step 4: Selezione delle top N features
    selected = select_features_with_model(df, candidates, targets=["High", "Low"], top_n=30)

    # Step 5: Aggiungi le base se mancano
    for f in base_features:
        if f not in selected:
            selected.append(f)

    # Step 6: Salva e logga
    save_selected_features(selected, args.data_folder, symbol)
    print(f"✅ Selected features ({len(selected)}): {selected}")
else:
    # Carica le features selezionate
    selected = load_selected_features(args.data_folder, symbol)
    print_log(f"\n✅ Features caricate dal file per {symbol}: {selected}")

SEQ_LEN = 60

# Prepara X e y non scalati, usando percentuali di movimento rispetto all'Open
X = df[selected].values
# --- Δ alti e bassi (sempre positivi, in % Open) -------------
delta_high = (df["High"].values - df["Open"].values) / df["Open"].values   # ≥ 0
delta_low  = (df["Open"].values - df["Low"].values)  / df["Open"].values   # ≥ 0

# --- direzione corpo  (−1…+1) ---------------------------------
delta_body = (df["Close"].values - df["Open"].values) / df["Open"].values  # segno

# --- stack colonne nella forma (n, 3) -------------------------
y_raw = np.column_stack([delta_high, delta_low, delta_body])

# Se è una coppia JPY, applica lo scaling
if is_jpy_pair:
    print_log(f"⚠️ Applicando scaling speciale per coppia JPY con fattore: {jpy_scale_factor}")
    y = y_raw / jpy_scale_factor
else:
    y = y_raw

# Crea sequenze non scalate
X_seq_raw, y_seq_raw = create_sequences(X, y, SEQ_LEN) 

# Rolling window (avanza di una fold alla volta)
window_size = int(len(X_seq_raw) * 0.6)
step_size = int((len(X_seq_raw) - window_size) / 3)
splits = []
for start in range(0, len(X_seq_raw) - window_size - step_size + 1, step_size):
    train_idx = np.arange(start, start + window_size)
    val_idx = np.arange(start + window_size, start + window_size + step_size)
    splits.append((train_idx, val_idx))

need_tuning = args.force_retrain or not os.path.exists(os.path.join(args.data_folder, f"models/best_hp_{symbol}.json"))

if need_tuning:
    print_log(f"\n🎯 Hyperparameter tuning con Hyperband per {symbol}...")

    best_val     = float("inf")   # loss migliore vista finora
    best_model   = None
    best_scalers = {"features": None, "target": None}
    best_hp      = None

    for fold, (train_idx, val_idx) in enumerate(splits):
        X_train, X_val = X_seq_raw[train_idx], X_seq_raw[val_idx]
        y_train, y_val = y_seq_raw[train_idx], y_seq_raw[val_idx]

        scaler_features = RobustScaler()
        X_train_scaled = scaler_features.fit_transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
        X_val_scaled = scaler_features.transform(X_val.reshape(-1, X_val.shape[2])).reshape(X_val.shape)

        scaler_y_high = RobustScaler(quantile_range=(5,95))
        scaler_y_low  = RobustScaler(quantile_range=(5,95))
        scaler_y_body = RobustScaler()                

        y_train_scaled = np.column_stack([
            scaler_y_high.fit_transform(y_train[:, [0]]),
            scaler_y_low .fit_transform(y_train[:, [1]]),
            scaler_y_body.fit_transform(y_train[:, [2]])
        ])
        y_val_scaled = np.column_stack([
            scaler_y_high.transform(y_val[:, [0]]),
            scaler_y_low .transform(y_val[:, [1]]),
            scaler_y_body.transform(y_val[:, [2]])
        ])

        num_features = X_train.shape[2]

        def build_model_with_features(hp):
            return build_model(hp, num_features)

        tuner = Hyperband(
            build_model_with_features,
            objective="val_loss",
            max_epochs=30,
            factor=3,
            directory="tuning",
            project_name=f"{symbol}_lstm_tune"
        )

        early_stop = EarlyStopping(monitor="val_loss", patience=4)

        tuner.search(
            X_train_scaled,
            [y_train_scaled[:, 0], y_train_scaled[:, 1], y_train_scaled[:, 2]],
            validation_data=(X_val_scaled, [y_val_scaled[:, 0], y_val_scaled[:, 1], y_val_scaled[:, 2]]),
            epochs=30,
            batch_size=32,
            callbacks=[early_stop],
            verbose=2,
            shuffle=False
        )

        best_model_fold = tuner.get_best_models(1)[0]
        best_hp_fold = tuner.get_best_hyperparameters(1)[0]

        val_loss = best_model_fold.evaluate(
            X_val_scaled,
            [y_val_scaled[:, 0], y_val_scaled[:, 1], y_val_scaled[:, 2]],
            verbose=0
        )[0]

        # --- Se è la migliore finora, salvala in memoria
        if val_loss < best_val:
            best_val   = val_loss

            best_model = tf.keras.models.clone_model(best_model_fold)
            best_model.set_weights(best_model_fold.get_weights())

            best_scalers["features"] = deepcopy(scaler_features)
            best_scalers["target"]   = {
                "high": deepcopy(scaler_y_high),
                "low":  deepcopy(scaler_y_low),
                "body": deepcopy(scaler_y_body),
            }
            best_hp = deepcopy(best_hp_fold)
            print_log(f"✅ Migliori iperparametri trovati: {best_hp.values}")

        os.makedirs(os.path.join(args.data_folder, "models"), exist_ok=True)

    print_log(f"🏆  Best validation loss over all folds: {best_val:.6f}")

    models_dir        = os.path.join(args.data_folder, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path        = os.path.join(models_dir, f"model_{symbol}.keras")
    scaler_feat_path  = os.path.join(models_dir, f"scaler_features_{symbol}.joblib")
    scaler_y_path     = os.path.join(models_dir, f"scaler_y_{symbol}.joblib")
    hp_path           = os.path.join(models_dir, f"best_hp_{symbol}.json")

    best_model.save(model_path, save_format="keras")
    joblib.dump(best_scalers["features"], scaler_feat_path)
    joblib.dump(best_scalers["target"], scaler_y_path)

    best_hp_values = best_hp.values
    best_hp_values["num_features"] = best_model.input_shape[-1]   
    with open(hp_path, "w") as fp:
        json.dump(best_hp_values, fp, indent=2)

    model = best_model

else:
    print_log(f"\n♻️  Caricamento modello e iperparametri salvati per {symbol}")
    model, scaler_features, scaler_y_high, scaler_y_low, scaler_y_body, hp_values = load_best_model_and_hp(symbol, args.data_folder, SEQ_LEN)

# ✅ In ogni caso: addestramento finale su tutti i dati
X_all = df[selected].values
delta_high_all = (df["High"].values - df["Open"].values) / df["Open"].values
delta_low_all  = (df["Open"].values - df["Low"].values)  / df["Open"].values
delta_body_all = (df["Close"].values - df["Open"].values) / df["Open"].values

y_all_raw = np.column_stack([delta_high_all, delta_low_all, delta_body_all])
y_all = y_all_raw / jpy_scale_factor if is_jpy_pair else y_all_raw

lr_final = (
    best_hp.get("lr") if need_tuning else hp_values.get("lr", 1e-3)
)

model.compile(
    optimizer=Adam(learning_rate=lr_final),
    loss={
        'high_output': quantile_loss(0.80),
        'low_output' : quantile_loss(0.20),
        'body_output': 'mse',
    },
    loss_weights={'high_output': 1.0, 'low_output': 1.0, 'body_output': 0.5}
)

# Ricarica il modello ottimizzato

X_seq_all, y_seq_all = create_sequences(X_all, y_all, SEQ_LEN)

scaler_features.fit(X_seq_all.reshape(-1, X_seq_all.shape[2]))
scaler_y_high.fit(y_seq_all[:, [0]])
scaler_y_low .fit(y_seq_all[:, [1]])
scaler_y_body.fit(y_seq_all[:, [2]])

X_scaled = (
    scaler_features.transform(X_seq_all.reshape(-1, X_seq_all.shape[2]))
                  .reshape(X_seq_all.shape)
)
y_scaled = np.column_stack([
    scaler_y_high.transform(y_seq_all[:, [0]]),
    scaler_y_low .transform(y_seq_all[:, [1]]),
    scaler_y_body.transform(y_seq_all[:, [2]])
])

joblib.dump(
    {"high": scaler_y_high, "low": scaler_y_low, "body": scaler_y_body},
    os.path.join(args.data_folder, f"models/scaler_y_{symbol}.joblib")
)
joblib.dump(scaler_features, os.path.join(args.data_folder, f"models/scaler_features_{symbol}.joblib"))

early_stop = EarlyStopping(monitor="loss", patience=5)

model.fit(
    X_scaled,
    [y_scaled[:, 0], y_scaled[:, 1], y_scaled[:, 2]],
    epochs=30,
    batch_size=32,
    callbacks=[early_stop],
    verbose=2,
    shuffle=False
)

# Per la predizione, usa gli ultimi SEQ_LEN punti dati
last_sequence = X[-SEQ_LEN:] 

# ✅ Applica lo scaler in un colpo solo
last_sequence_scaled = scaler_features.transform(last_sequence)

# ✅ Logging per debug (ultimo punto della sequenza)
print_log(f"\n🧾 Ultima sequenza (non scalata):\n{dict(zip(selected, last_sequence[-1]))}")
print_log(f"\n📉 Ultima sequenza SCALATA (input al modello):\n{dict(zip(selected, last_sequence_scaled[-1]))}")

last_sequence_scaled = last_sequence_scaled.reshape(1, SEQ_LEN, len(selected))

last_price = args.next_open if args.next_open else df.iloc[-1]["Close"]
# Esegui la previsione con il nuovo modello
y_pred_high_sc, y_pred_low_sc, y_pred_body_sc = model.predict(last_sequence_scaled)
delta_high = scaler_y_high.inverse_transform(y_pred_high_sc)[0][0]
delta_low  = scaler_y_low .inverse_transform(y_pred_low_sc )[0][0]
delta_body = scaler_y_body.inverse_transform(y_pred_body_sc)[0][0]

# Applica scala JPY se necessario
if is_jpy_pair:
    delta_high = delta_high * jpy_scale_factor
    delta_low = delta_low * jpy_scale_factor
    delta_body = delta_body * jpy_scale_factor

open_price = last_price          
if delta_body >= 0:              # candela verde
    w_high, w_low = 1.0, 0.2
else:                            # candela rossa
    w_high, w_low = 0.2, 1.0

# predicted_high = open_price * (1 + delta_high * w_high)
# predicted_low  = open_price * (1 - delta_low  * w_low)
predicted_high = open_price * (1 + delta_high)
predicted_low  = open_price * (1 - delta_low)

print_log(f"✅ Open: {last_price:.5f}")
print_log(f"✅ Predicted High: {predicted_high:.5f}, Predicted Low: {predicted_low:.5f}")

# === Calcolo probabilità su livelli di prezzo basati su pips ===
pip_size = pip_values.get(symbol.upper(), 0.0001)

# Parametri
pip_levels = np.arange(-40, 42, 2)
price_levels = last_price + pip_levels * pip_size

# Range previsto
low, high = predicted_low, predicted_high
midpoint = (low + high) / 2
softness = 1.5

# Funzione di probabilità "touch" migliorata e asimmetrica
def touch_prob(level):
    if low <= level <= high:
        return 1.0
    else:
        if level < low:
            ref = low
            side_span = abs(midpoint - low)
        else:
            ref = high
            side_span = abs(high - midpoint)

        distance = abs(level - ref)
        return np.exp(-distance / (side_span / softness + 1e-9))

# Calcolo e normalizzazione
probs = [touch_prob(lvl) for lvl in price_levels]
probs = np.array(probs)
probs /= probs.max()  # scala 0–1
probs *= 100          # scala 0–100
probs = np.clip(probs, 0, 100)

# Dizionario finale
prob_dict = {str(int(p)): round(float(prob), 2) for p, prob in zip(pip_levels, probs)}

# Decisione: buy/sell/wait
threshold_pips = 14 if "JPY" in symbol.upper() else 7
threshold_value = threshold_pips * pip_size

max_safe_pips = 50
max_safe_value = max_safe_pips * pip_size

dist_high = abs(predicted_high - last_price)
dist_low = abs(predicted_low - last_price)

# 1. Se la differenza tra le due distanze è troppo piccola → wait
if abs(dist_high - dist_low) < threshold_value:
    decision = "wait"
# 2. Se il lato alto è nettamente più lontano → buy
elif dist_high > dist_low:
    if dist_low > max_safe_value:
        decision = "wait"
    else:
        decision = "buy"
# 3. Se il lato basso è nettamente più lontano → sell
else:
    if dist_high > max_safe_value:
        decision = "wait"
    else:
        decision = "sell"

# Data prevista per la barra (solo se Open fornito)
predicted_dt = df["DateTime"].max() + pd.Timedelta(days=1)

# Creazione struttura JSON finale
prediction_result = {
    "open": args.next_open,
    "predicted_high": round(predicted_high, decimal_precision),
    "predicted_low": round(predicted_low, decimal_precision),
    "predictions": prob_dict,
    "decision": decision,
    "next_datetime": str(predicted_dt)
}

# Salvataggio file JSON
os.makedirs("predictions", exist_ok=True)
with open(f"predictions/predictions_{symbol.upper()}.json", "w") as f:
    json.dump(prediction_result, f, indent=2)

print_log(f"\n📊 Previsioni salvate in: predictions/predictions_{symbol.upper()}.json")