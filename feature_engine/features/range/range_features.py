from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Dict, List

import numpy as np
import pandas as pd


@dataclass
class RangeFeatureConfig:
    
    """
    Configuration for dynamic range feature extraction. All configuration values are subject to change.

    windows:
    
        Rolling windows used to describe short, medium, and longer range behavior.

    atr_window:
    
        Average True Range measures market volatility. It was "Developed by J. Welles Wilder Jr., it calculates the average 
        range of an asset's price movement over a specified period (typically 14 days), taking price gaps and limits into 
        account" (https://en.wikipedia.org/wiki/Average_true_range).

    zone_pct:
    
        Percentage of the rolling range width used to define upper/lower boundary zones. The reason this is needed is because we must assume that
        the high and low of a range is not respected by price. A new high or low can be formed at any time. Therefore I establish a small zone 
        beyond the concrete high/low of the range. I will respect this assumption even though this classifier is not meant to predict future 
        behavior.
        
        For example... zone_pct = 0.15 means: upper zone = top 15% of the range | lower zone = bottom 15% of the range

    slope_window:
    
        Window used when measuring change in feature values since they are not static.

    min_periods_ratio:
    
        Required fraction, or data, of a rolling window before initiating the calculation of a feature.
        
        For example: window=100, min_periods_ratio=0.8 means at least 80 candles needed. 

    eps:
        A small value to be used for avoiding division by 0. 
    """

    windows: tuple[int, ...] = (20, 50, 100)
    
    atr_window: int = 14
    
    zone_pct: float = 0.15
    
    slope_window: int = 5
    
    min_periods_ratio: float = 0.8
    
    eps: float = 1e-12



class RangeFeatureExtractor:

    """

    Extracts dynamic range-behavior features.

    This class does NOT predict.

    This class does NOT produce buy/sell signals.

    This class describes how the current rolling market window behaves.

    Feature families:

        1. Range geometry

        2. ATR-normalized width

        3. Position inside range

        4. Directional efficiency

        5. Boundary zones and touch counts

        6. Touch balance or two-sided activity

        7. Midpoint rotation

        8. Wick rejection near boundaries

        9. Slope / flatness

        10. Volatility compression

        11. Lifecycle/change features

        12. Multi-window comparison features

    """

    def __init__(
        self,
        config: Optional[RangeFeatureConfig] = None,
        *,
        windows: Optional[Iterable[int]] = None,
        atr_window: Optional[int] = None,
        zone_pct: Optional[float] = None,
        slope_window: Optional[int] = None,
    ) -> None:

        self.config = config or RangeFeatureConfig()

        if windows is not None:

            self.config.windows = tuple(windows)

        if atr_window is not None:

            self.config.atr_window = atr_window

        if zone_pct is not None:

            self.config.zone_pct = zone_pct

        if slope_window is not None:

            self.config.slope_window = slope_window

        self._validate_config()

    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        """

        Main feature extraction method.

        Inputs df: Normalized OHLC dataframe. In this case data from MarketNormalizationEngine.

        Returns pd.DataFrame: Original dataframe plus range-behavior features.

        """

        """
        TODO: 
        
        Z Score = ((X - u) / sigma) where X is the most recent data point, u is the rolling average (mean) over the defined lookback window, and 
        sigma is the rolling standard deviation over the same window.
        """
        
        data = self._standardize_ohlc_columns(df) #Ensures data from MarketNormalizationEngine is compatible.

        data = data.copy()

        self._validate_ohlc_data(data)

        data = self._add_candle_features(data)

        data = self._add_atr(data)

        for window in self.config.windows:

            data = self._add_range_geometry_features(data, window)

            data = self._add_directional_efficiency_features(data, window)

            data = self._add_boundary_touch_features(data, window)

            data = self._add_rotation_features(data, window)

            data = self._add_wick_rejection_features(data, window)

            data = self._add_slope_flatness_features(data, window)

            data = self._add_lifecycle_features(data, window)

        data = self._add_multi_window_comparison_features(data)

        return data



    # ---------------------------------------------------------------------
    #                          Helper Functions
    # ---------------------------------------------------------------------

    def _standardize_ohlc_columns(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Makes this extractor compatible with the output from my MarketNormalizationEngine.
        """

        data = df.copy()

        lower_map: Dict[str, str] = {col.lower(): col for col in data.columns}

        rename_map = {}

        column_aliases = {
            "timestamp": ["timestamp", "date", "datetime", "time"],
            "open": ["open", "o"],
            "high": ["high", "h"],
            "low": ["low", "l"],
            "close": ["close", "c"],
            "volume": ["volume", "vol", "tick_volume"],
        }

        for standard_name, aliases in column_aliases.items():

            for alias in aliases:

                if alias in lower_map:

                    rename_map[lower_map[alias]] = standard_name

                    break

        data = data.rename(columns=rename_map)

        required = ["open", "high", "low", "close"]

        missing = [col for col in required if col not in data.columns]

        if missing:

            raise ValueError(

                f"Missing required OHLC columns: {missing}. "

                f"Available columns: {list(df.columns)}. "

                "Expected columns like Open, High, Low, Close or open, high, low, close."

            )

        if "timestamp" in data.columns:

            data["timestamp"] = pd.to_datetime(data["timestamp"])

            data = data.sort_values("timestamp").reset_index(drop=True)

        for col in ["open", "high", "low", "close"]:

            data[col] = pd.to_numeric(data[col], errors="coerce")

        if "volume" in data.columns:

            data["volume"] = pd.to_numeric(data["volume"], errors="coerce")

        return data