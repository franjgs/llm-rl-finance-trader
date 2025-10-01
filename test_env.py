#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 18:40:19 2025

@author: fran
"""

import pytest
import pandas as pd
import numpy as np
from src.trading_env import TradingEnv  # Importa desde src/

@pytest.fixture
def sample_df():
    """Crea un DataFrame de prueba"""
    return pd.DataFrame({
        'open': [100, 101, 102],
        'high': [110, 111, 112],
        'low': [90, 91, 92],
        'close': [105, 106, 107],
        'volume': [1000, 1100, 1200],
        'sentiment': [0.5, -0.3, 0.1]
    })

def test_env_init(sample_df):
    """Testea inicialización del entorno"""
    env = TradingEnv(sample_df)
    assert env.balance == 10000
    assert env.shares_held == 0
    assert env.net_worth == 10000
    assert env.current_step == 0
    assert env.observation_space.shape == (11,)  # 5 precios + sentiment + balance/net_worth/cost + etc.

def test_env_step_buy(sample_df):
    """Testea acción de compra"""
    env = TradingEnv(sample_df)
    action = np.array([0.5, 0.2])  # Buy, 20% del monto posible
    obs, reward, terminated, _, _ = env.step(action)
    assert env.shares_held > 0
    assert env.balance < 10000
    assert not terminated

def test_env_step_sell(sample_df):
    """Testea acción de venta"""
    env = TradingEnv(sample_df)
    env.shares_held = 10  # Simula tenencia
    env.balance = 5000
    action = np.array([1.5, 0.2])  # Sell, 20% del monto
    obs, reward, terminated, _, _ = env.step(action)
    assert env.shares_held < 10
    assert env.balance > 5000
    assert not terminated

def test_env_reset(sample_df):
    """Testea reset del entorno"""
    env = TradingEnv(sample_df)
    obs, _ = env.reset()
    assert env.balance == 10000
    assert env.shares_held == 0
    assert env.current_step == 0