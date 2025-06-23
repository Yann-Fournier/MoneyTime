import ta
import numpy as np
import pandas as pd
import mplfinance as mpf


# Indicateurs techniques ------------------------------------------------------------------------------------------------------
def rsi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fonction pour calculer l'indicateur RSI (Relative Strength Index) et sa moyenne sur 14 unitées de temps.

    Args:
        df (pd.Dataframe): dataframe contenant les données de prix avec une colonne 'close'.

    Returns:
        pd.Dataframe: retourne le dataframe avec les colonnes 'rsi' et 'rsima' ajoutées.
    """
    df["rsi"] = ta.momentum.rsi(df['close'], 14) # rsi avec une periode de 14 unitées de temps
    df['rsima'] = df['rsi'].rolling(14).mean() # une moyenne des 14 dernières unitées de temps
    return df

def macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fonction pour calculer l'indicateur MACD (Moving Average Convergence Divergence) et ses composantes.

    Args:
        df (pd.Dataframe): dataframe contenant les données de prix avec une colonne 'close'.

    Returns:
        pd.Dataframe: retourne le dataframe avec les colonnes 'macd', 'macd_diff' et 'macd_signal' ajoutées.
    """
    # macd avec une période lente de 26 unitées de temps et une période rapide de 12 unitées de temps
    df["macd"] = ta.trend.macd(close=df["close"], window_slow=26, window_fast=12)
    # macd différence avec une période lente de 26 unitées de temps et une période rapide de 12 unitées de temps
    df["macd_diff"] = ta.trend.macd_diff(close=df["close"], window_slow=26, window_fast=12)
    # macd signal avec une période lente de 26 unitées de temps et une période rapide de 12 unitées de temps
    df["macd_signal"] = ta.trend.macd_signal(close=df["close"], window_slow=26, window_fast=12, window_sign=9, fillna=False)
    return df

def stochastique(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fonction pour calculer l'indicateur Stochastique et ses moyennes.

    Args:
        df (pd.Dataframe): dataframe contenant les données de prix avec une colonne 'close'.

    Returns:
        pd.Dataframe: retourne le dataframe avec les colonnes 'stock_%K' et 'stock_%D' ajoutées.
    """
    # Stochastique avec une periode de 14 unitées de temps
    df["stoch_%K"] = ta.momentum.stoch(high=df["high"], low=df["low"], close=df["close"], window= 14,smooth_window=3, fillna=False)
    # moyenne des trois dernières unitées de temps du Stochastique
    df["stoch_%D"] = df['stoch_%K'].rolling(3).mean()
    return df

def bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fonction pour calculer les bandes de Bollinger et les ajouter au dataframe.

    Args:
        df (pd.Dataframe): dataframe contenant les données de prix avec une colonne 'close'.

    Returns:
        pd.Dataframe: retourne le dataframe avec les colonnes 'lower_band', 'higher_band' et 'ma_band' ajoutées.
    """
    # Bandes de Bollinger effectuer sur une période de 20 unitées de temps
    bol_band = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2.25)
    df["lower_band"] = bol_band.bollinger_lband()
    df["higher_band"] = bol_band.bollinger_hband()
    df["ma_band"] = bol_band.bollinger_mavg()
    return df

def moyenne_mobile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fonction pour calculer les moyennes mobiles et les ajouter au dataframe.

    Args:
        df (pd.Dataframe): dataframe contenant les données de prix avec une colonne 'close'.

    Returns:
        pd.Dataframe: retourne le dataframe avec les colonnes 'sma9' et 'sma21' ajoutées.
    """
    df["sma9"] = df['close'].rolling(9).mean() # moyenne des 9 dernières unitées de temps du prix de fermeture de session
    df["sma21"] = df['close'].rolling(21).mean() # moyenne des 21 dernières unitées de temps du prix de fermeture de session
    return df


# Graphiques ------------------------------------------------------------------------------------------------------
def simple_plot(df: pd.DataFrame, val: str) -> None: # création d'un graphique simple (que la valeur)
    """
    Fonction pour tracer un graphique simple avec les données de prix.

    Args:
        df (pd.Dataframe): dataframe contenant les données de prix.
        val (str): titre du graphique.
    """
    df_plot = df.copy().iloc[-150:] # on prend les 150 dernières valeurs du dataset.
    s = mpf.make_mpf_style(base_mpf_style='charles', rc={'font.size': 6}) # on indique le style du graphique
    fig = mpf.figure(2, figsize=(20, 15), style=s) # création de la figure
    ax1 = fig.add_subplot(2,1,1, title=val) # ajout d'un graphique
    mpf.plot(df_plot, type='candle', ax=ax1) # ajout des données dans le graphique
    ax1.yaxis.set_label_position('left')  # positionnement du label des ordonnées à gauche (style)
    ax1.yaxis.tick_left() # positionnement de l'axe des ordonnées à gauche (style)
    
    
def plot_rsi(df: pd.DataFrame, val: str) -> None:
    """
    Fonction pour tracer un graphique avec l'indicateur RSI (Relative Strength Index).

    Args:
        df (pd.Dataframe): dataframe contenant les données de prix avec des colonnes 'rsi' et 'rsima'.
        val (str): titre du graphique.
    """
    df_plot = df.copy().iloc[-150:] # on prend les 150 dernières valeurs du dataset.
    s = mpf.make_mpf_style(base_mpf_style='charles', rc={'font.size': 6}) # on indique le style du graphique
    fig1 = mpf.figure(1, figsize=(20, 15), style=s) # création de la figure
    ax1 = fig1.add_subplot(2,1,1,title=val) # ajout d'un graphique
    ax2 = fig1.add_subplot(2,1,2, sharex=ax1, title="RSI") # ajout d'un deuxieme graphique
    ap0 = [ # Les sous graphiques doivent être stockés dans un tableau. c'est le parametre 'ax' qui defini sur quelle graphique ils sont ajouter
        mpf.make_addplot(df_plot["rsi"], color='purple', panel=0, ylabel='Points', ax=ax2), # ajout d'un sous graphique sur le deuxieme graphique
        mpf.make_addplot(df_plot["rsima"], color='blue', panel=0,ax=ax2) # ajout d'un sous graphique sur le deuxieme graphique
    ]
    mpf.plot(df_plot, type='candle', ax=ax1, addplot=ap0) # ajout des données dans le graphique + ajout des sous graphiques
    ax1.yaxis.set_label_position('left') # positionnement du label des ordonnées à gauche (style)
    ax1.yaxis.tick_left() # positionnement de l'axe des ordonnées à gauche (style)
    ax2.axhline(30, color='black', linestyle='--') # ajout d'une ligne pointillé sur le deuxieme graphique sur l'ordonnée 30
    ax2.axhline(50, color='black', linestyle='--') # ajout d'une ligne pointillé sur le deuxieme graphique sur l'ordonnée 50
    ax2.axhline(70, color='black', linestyle='--') # ajout d'une ligne pointillé sur le deuxieme graphique sur l'ordonnée 70 
    
    
def plot_macd(df: pd.DataFrame, val: str) -> None:
    """
    Fonction pour tracer un graphique avec l'indicateur MACD (Moving Average Convergence Divergence).

    Args:
        df (pd.Dataframe): dataframe contenant les données de prix avec des colonnes 'macd', 'macd_diff' et 'macd_signal'.
        val (str): titre du graphique.
    """
    df_plot = df.copy().iloc[-150:] # on prend les 150 dernières valeurs du dataset.
    s = mpf.make_mpf_style(base_mpf_style='charles', rc={'font.size': 10})# on indique le style du graphique
    fig2 = mpf.figure(2, figsize=(20, 15), style=s) # création de la figure
    ax1 = fig2.add_subplot(2,1,1, title=val) # ajout d'un graphique
    ax2 = fig2.add_subplot(2,1,2, sharex=ax1, title="MACD") # ajout d'un deuxieme graphique
    ap0 = [ # Les sous graphiques doivent être stockés dans un tableau. c'est le parametre 'ax' qui defini sur quelle graphique ils sont ajouter
        mpf.make_addplot(df_plot["macd"]/10, color='blue', panel=0, ylabel='Points', ax=ax2), # ajout d'un sous graphique sur le deuxieme graphique
        mpf.make_addplot(df_plot["macd_signal"]/10, color='orange', panel=0, ax=ax2), # ajout d'un sous graphique sur le deuxieme graphique
        mpf.make_addplot(df_plot["macd_diff"]/10, panel=0, ax=ax2, type='bar', color='lightblue') # ajout d'un sous graphique sur le deuxieme graphique
    ]
    mpf.plot(df_plot, type='candle', ax=ax1, addplot=ap0) # ajout des données dans le graphique + ajout des sous graphiques
    ax1.yaxis.set_label_position('left') # positionnement du label des ordonnées à gauche (style)
    ax1.yaxis.tick_left() # positionnement de l'axe des ordonnées à gauche (style)
    ax2.axhline(0, color='black', linestyle='--') # ajout d'une ligne pointillé sur le deuxieme graphique sur l'ordonnée 0


def plot_stochastique(df: pd.DataFrame, val: str) -> None:
    """
    Fonction pour tracer un graphique avec l'indicateur Stochastique.

    Args:
        df (pd.Dataframe): dataframe contenant les données de prix avec des colonnes 'stoch_%K' et 'stoch_%D'.
        val (str): titre du graphique.
    """
    df_plot = df.copy().iloc[-150:] # on prend les 150 dernières valeurs du dataset.
    s = mpf.make_mpf_style(base_mpf_style='charles', rc={'font.size': 10}) # on indique le style du graphique
    fig3 = mpf.figure(3, figsize=(20, 15), style=s) # création de la figure
    ax1 = fig3.add_subplot(2,1,1, title=val) # ajout d'un graphique
    ax2 = fig3.add_subplot(2,1,2, sharex=ax1, title="Stochastic") # ajout d'un deuxieme graphique
    ap0 = [ # Les sous graphiques doivent être stockés dans un tableau. c'est le parametre 'ax' qui defini sur quelle graphique ils sont ajouter
        mpf.make_addplot(df_plot["stoch_%K"], color='blue', panel=0, ylabel='Points', ax=ax2), # ajout d'un sous graphique sur le deuxieme graphique
        mpf.make_addplot(df_plot["stoch_%D"], color='orange', panel=0, ax=ax2) # ajout d'un sous graphique sur le deuxieme graphique
    ]
    mpf.plot(df_plot, type='candle', ax=ax1, addplot=ap0) # ajout des données dans le graphique + ajout des sous graphiques
    ax1.yaxis.set_label_position('left') # positionnement du label des ordonnées à gauche (style)
    ax1.yaxis.tick_left() # positionnement de l'axe des ordonnées à gauche (style)
    ax2.axhline(20, color='black', linestyle='--') # ajout d'une ligne pointillé sur le deuxieme graphique sur l'ordonnée 20
    ax2.axhline(50, color='black', linestyle='--') # ajout d'une ligne pointillé sur le deuxieme graphique sur l'ordonnée 30
    ax2.axhline(80, color='black', linestyle='--') # ajout d'une ligne pointillé sur le deuxieme graphique sur l'ordonnée 80


def plot_bollinger_bands(df: pd.DataFrame, val: str) -> None:
    """
    Fonction pour tracer un graphique avec les bandes de Bollinger.

    Args:
        df (pd.Dataframe): dataframe contenant les données de prix avec des colonnes 'lower_band', 'higher_band' et 'ma_band'.
        val (str): titre du graphique.
    """
    df_plot = df.copy().iloc[-150:] # on prend les 150 dernières valeurs du dataset.
    s = mpf.make_mpf_style(base_mpf_style='charles', rc={'font.size': 10}) # on indique le style du graphique
    fig4 = mpf.figure(4, figsize=(20, 15), style=s) # création de la figure
    ax1 = fig4.add_subplot(2,1,1, title=val + " / Bollinger Bands") # ajout d'un graphique
    ap0 = [ # Les sous graphiques doivent être stockés dans un tableau. c'est le parametre 'ax' qui defini sur quelle graphique ils sont ajouter
        mpf.make_addplot(df_plot["lower_band"], color='blue', panel=0, ax=ax1), # ajout d'un sous graphique sur le graphique
        mpf.make_addplot(df_plot["higher_band"], color='blue', panel=0, ax=ax1), # ajout d'un sous graphique sur le graphique
        mpf.make_addplot(df_plot["ma_band"], color='orange', panel=0, ax=ax1) # ajout d'un sous graphique sur le graphique
    ]
    mpf.plot(df_plot, type='candle', ax=ax1, addplot=ap0) # ajout des données dans le graphique + ajout des sous graphiques
    ax1.yaxis.set_label_position('left') # positionnement du label des ordonnées à gauche (style)
    ax1.yaxis.tick_left() # positionnement de l'axe des ordonnées à gauche (style)


def plot_moyenne_mobile(df: pd.DataFrame, val: str) -> None:
    """
    Fonction pour tracer un graphique avec les moyennes mobiles.

    Args:
        df (pd.Dataframe): dataframe contenant les données de prix avec des colonnes 'sma9' et 'sma21'.
        val (str): titre du graphique.
    """
    df_plot = df.copy().iloc[-150:] # on prend les 150 dernières valeurs du dataset.
    s = mpf.make_mpf_style(base_mpf_style='charles', rc={'font.size': 10}) # on indique le style du graphique
    fig5 = mpf.figure(5, figsize=(20, 15), style=s) # création de la figure
    ax1 = fig5.add_subplot(2,1,1, title=val + " / Moyenne Mobile") # ajout d'un graphique
    ap0 = [ # Les sous graphiques doivent être stockés dans un tableau. c'est le parametre 'ax' qui defini sur quelle graphique ils sont ajouter
        mpf.make_addplot(df_plot["sma9"], color='blue', panel=0, ylabel='Points', ax=ax1), # ajout d'un sous graphique sur le graphique
        mpf.make_addplot(df_plot["sma21"], color='orange', panel=0, ax=ax1), # ajout d'un sous graphique sur le graphique
    ]     
    mpf.plot(df_plot, type='candle', ax=ax1, addplot=ap0) # ajout des données dans le graphique + ajout des sous graphiques
    ax1.yaxis.set_label_position('left') # positionnement du label des ordonnées à gauche (style)
    ax1.yaxis.tick_left() # positionnement de l'axe des ordonnées à gauche (style)
    

# Add indicators to dataframe ------------------------------------------------------------------------------------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fonction pour ajouter des indicateurs techniques au dataframe.

    Args:
        df (pd.Dataframe): dataframe contenant les données de prix avec des colonnes 'open', 'close', 'high', 'low'.

    Returns:
        pd.Dataframe: retourne le dataframe avec les indicateurs techniques ajoutés.
    """
    df = rsi(df) # ajout de l'indicateur RSI
    df = macd(df) # ajout de l'indicateur MACD
    df = stochastique(df) # ajout de l'indicateur Stochastique
    df = bollinger_bands(df) # ajout des bandes de Bollinger
    df = moyenne_mobile(df) # ajout des moyennes mobiles
    df['Avg_day'] = ((df['close'] + df['open'] + df['high'] + df['low']) / 4).round(2) # calcul de la moyenne du prix de la bougie (open, close, high, low)
    df['Avg_corps'] = ((df['close'] + df['open']) / 2).round(2) # calcul de la moyenne du prix du corps de la bougie (open, close)
    df['Avg_meches'] = ((df['high'] + df['low']) / 2).round(2) # calcul de la moyenne du prix des mèches de la bougie (high, low)
    df['Pourc_Price_Evol_14d'] = ((100 * df['Avg_day'].shift(-14) / df['Avg_day']) - 100).round(2) # calcul de l'évolution du prix sur 14 jours en pourcentage
    df['Label'] = np.where(df['Pourc_Price_Evol_14d'] > 5, 1,np.where(df['Pourc_Price_Evol_14d'] < -5, -1, 0)) # création de la colonne Label qui indique si le prix a augmenté de plus de 5% (1), diminué de plus de 5% (-1) ou est resté stable (0)
    df = df.dropna() # suppression des lignes avec des valeurs manquantes
    #df = df.reset_index() # réinitialisation de l'index du dataframe
    return df


