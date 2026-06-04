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

    zscore_windows:

        Window used when measuring rolling z-score
        
    zscore_clip:

        Used to stop an outlier from distorting the classification model. If I set the clip to 5, then any standard deviation above or below the
        clip will become the clip. 
        
    min_periods_ratio:
    
        Required fraction, or data, of a rolling window before initiating the calculation of a feature.
        
        For example: window=100, min_periods_ratio=0.8 means at least 80 candles needed. 

    eps:
        A small value to be used for avoiding division by 0. 
    """

    windows: tuple[int, ...] = (20, 50, 100)
    
    atr_window: int = 14

    atr_method: str = "wilder"
    
    zone_pct: float = 0.15
    
    slope_window: int = 5

    #Note that it is bigger than main window. I am asking "Is behavior in window unusual from previous data?
    zscore_windows: tuple[int, ...] = (100, 250) 

    zscore_clip: float = 5.0
    
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
        atr_method: Optional[int] = None,
        zone_pct: Optional[float] = None,
        slope_window: Optional[int] = None,
        zscore_windows: Optional[Iterable[int]] = None,
        zscore_clip: Optional[float] = None) -> None:

        
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

        #Now that base features are added we can calculate rolling z-score for each one
        data = self._add_rolling_zscores(data)
        
        return data

    

    # ---------------------------------------------------------------------
    #                          Rolling Z-SCORE
    # ---------------------------------------------------------------------

    def _add_rolling_zscores(self, df: pd.DataFrame) -> pd.DataFrame:
    
    """
    Adds rolling z-scores.

    Z Score = ((X - u) / σ) where X is the most recent data point, u is the rolling average (mean) over the defined lookback window, and 
    sigma is the rolling standard deviation over the same window.
        
    The rolling mean/std are shifted by 1 so the current row is compared
    only against prior feature history. This avoids the current value 
    influencing its own normalization.
    """

        for base_col in self._zscore_feature_columns(df):
        
            if base_col not in df.columns:
                
                continue

            x = df[base_col].astype(float)

            for z_window in self.config.zscore_windows:
                
                min_periods = self._min_periods(z_window)

                rolling_mean = (x.rolling(window=z_window, min_periods=min_periods).mean().shift(1))

                rolling_std = (x.rolling(window=z_window, min_periods=min_periods).std(ddof=0).shift(1))

                z_col = f"{base_col}_z{z_window}"

                safe_std = rolling_std.where( rolling_std > self.config.eps, np.nan)

                z = (x - rolling_mean) / safe_std

                df[z_col] = z.clip(lower=-self.config.zscore_clip, upper=self.config.zscore_clip)

        return df


    # ---------------------------------------------------------------------
    #                            Validation
    # ---------------------------------------------------------------------

    def _validate_config(self) -> None:
        
        if not self.config.windows:
            
            raise ValueError("windows cannot be empty.")

        if any(window <= 1 for window in self.config.windows):
            
            raise ValueError("All windows must be greater than 1.")

        if self.config.atr_window <= 1:
            
            raise ValueError("atr_window must be greater than 1.")

        if not 0.01 <= self.config.zone_pct <= 0.45:
            
            raise ValueError("zone_pct should usually be between 0.01 and 0.45.")

        if self.config.slope_window <= 1:
            
            raise ValueError("slope_window must be greater than 1.")

        if not self.config.zscore_windows:
            
            raise ValueError("zscore_windows cannot be empty.")

        if any(window <= 2 for window in self.config.zscore_windows):
            
            raise ValueError("All zscore_windows must be greater than 2.")

        if self.config.zscore_clip <= 0:
            
            raise ValueError("zscore_clip must be positive.")

        if not 0.1 <= self.config.min_periods_ratio <= 1.0:
        
            raise ValueError("min_periods_ratio must be between 0.1 and 1.0.")

        if self.config.atr_method not in {"wilder"}:

            raise ValueError("atr_method must be 'wilder'.")


    # ---------------------------------------------------------------------
    #                          Slope Feature(s)
    # ---------------------------------------------------------------------
            
    @staticmethod
    def _theil_sen_slope(values: np.ndarray) -> float:
        """
        Robust slope estimator.

        Uses the median of all pairwise slopes.
        More resistant to outlier candles than ordinary least-squares slope that uses linear regression.

        For rolling windows like 20, 50, 100 this is acceptable, but for very large
        windows it can become expensive since it is O(n^2).

        TODO: Upgrade to a multithreaded implementation.
        """

        values = np.asarray(values, dtype=float)

        if len(values) < 2:
            
            return np.nan

        
        if np.isnan(values).any():
            
            return np.nan

        
        slopes = []

        for i in range(len(values) - 1):
            
            for j in range(i + 1, len(values)):
                
                denominator = j - i

                if denominator == 0:
                    
                    continue

                slopes.append((values[j] - values[i]) / denominator)

        
        if not slopes:
            
            return np.nan

        
        return float(np.median(slopes))

    

    def _add_slope_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        
        """
        Adds slope/flatness features for a rolling close window.

        Why? It describes whether the current window is flat/balanced or tilted/directional.

        Why both slopes? OLS slope reacts strongly to outlier closes. Theil-Sen slope is more resistant to outlier candles.
        The difference between them can describe outlier sensitivity.

        Features created for each window:
            
            close_slope_N
            close_slope_atr_N
            abs_close_slope_atr_N

            trendline_move_atr_N
            abs_trendline_move_atr_N

            robust_close_slope_N
            robust_close_slope_atr_N
            abs_robust_close_slope_atr_N

            robust_trendline_move_atr_N
            abs_robust_trendline_move_atr_N

            flatness_score_N
            slope_outlier_sensitivity_N
        """

        c = self.config
        min_periods = self._min_periods(window)
        atr_col = f"atr_{c.atr_window}"

        
        # ------------------------------------------------------------------
        # Ordinary least-squares slope
        # ------------------------------------------------------------------

        slope_col = f"close_slope_{window}"

        df[slope_col] = (df["close"].rolling(window=window, min_periods=min_periods).apply(self._linear_regression_slope, raw=True))

        #Per-candle OLS slope normalized by ATR
        df[f"close_slope_atr_{window}"] = (df[slope_col] / (df[atr_col] + c.eps))

        df[f"abs_close_slope_atr_{window}"] = (df[f"close_slope_atr_{window}"].abs())

        #Total fitted OLS movement across the whole window, normalized by ATR
        df[f"trendline_move_atr_{window}"] = ((df[slope_col] * (window - 1)) / (df[atr_col] + c.eps))

        df[f"abs_trendline_move_atr_{window}"] = (df[f"trendline_move_atr_{window}"].abs())


        # ------------------------------------------------------------------
        # Theil-Sen slope
        # ------------------------------------------------------------------

        robust_slope_col = f"robust_close_slope_{window}"

        df[robust_slope_col] = (df["close"].rolling(window=window, min_periods=min_periods).apply(self._theil_sen_slope, raw=True))

        # Per-candle robust slope normalized by ATR.
        df[f"robust_close_slope_atr_{window}"] = (df[robust_slope_col] / (df[atr_col] + c.eps))

        df[f"abs_robust_close_slope_atr_{window}"] = (df[f"robust_close_slope_atr_{window}"].abs())

        # Total fitted robust movement across the whole window, normalized by ATR.
        df[f"robust_trendline_move_atr_{window}"] = ((df[robust_slope_col] * (window - 1)) / (df[atr_col] + c.eps))

        df[f"abs_robust_trendline_move_atr_{window}"] = (df[f"robust_trendline_move_atr_{window}"].abs())

        # ------------------------------------------------------------------
        # Flatness score
        # ------------------------------------------------------------------
          """
          Main flatness uses robust slope because it is less distorted by
          one-candle outliers or abnormal closes.
        
          flatness_score close to 1 = flatter / more range-like
          flatness_score close to 0 = more tilted / more directional
          """

        df[f"flatness_score_{window}"] = (1.0 / (1.0 + df[f"abs_robust_trendline_move_atr_{window}"])).clip(lower=0.0, upper=1.0)

        # ------------------------------------------------------------------
        # Outlier sensitivity
        # ------------------------------------------------------------------
          """
          If OLS and robust slope disagree a lot, the window may contain an
          outlier close, sweep, spike, or abnormal displacement.
          """

        df[f"slope_outlier_sensitivity_{window}"] = (df[f"abs_trendline_move_atr_{window}"] - df[f"abs_robust_trendline_move_atr_{window}"]).abs()


        return df



    # ---------------------------------------------------------------------
    #                                  ATR
    # ---------------------------------------------------------------------
    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Adds True Range and Wilder ATR.

        True Range: max( high - low, abs(high - previous_close), abs(low - previous_close))

        Wilder ATR: Uses exponential smoothing with alpha = 1 / atr_window.

        Why wilder? ATR is used as a normalization denominator for range features. So wilder smoothing is less jumpy
        than a simple rolling mean, which helps keep features like range_width_atr_N and trendline_move_atr_N more stable.

        """

        c = self.config

        atr_col = f"atr_{c.atr_window}"

        prev_close = df["close"].shift(1)

        true_range_components = pd.concat( [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1)

        
        df["true_range"] = true_range_components.max(axis=1)

        
        df[atr_col] = (df["true_range"].ewm(alpha=1.0 / c.atr_window,
                                            adjust=False,
                                            min_periods=self._min_periods(c.atr_window)).mean())

        
        return df



    # ---------------------------------------------------------------------
    #                           Window Comaparison
    # ---------------------------------------------------------------------
    def _safe_ratio(self, df: pd.DataFrame, numerator: str,  denominator: str, output: str, *, clip: float | None = 10.0) -> None:

        """
        Adds a safe ratio feature.

        Why? Raw ratios can explode when the denominator is ~ zero so I want to avoid creating misleading outliers.
        Try range_width_atr_20 / range_width_atr_50 as an example to understand this more.
        """

        if numerator not in df.columns or denominator not in df.columns:

            df[output] = np.nan

            return


        dividend = df[numerator].astype(float)

        divisor = df[denominator].astype(float)

        safe_divisor = divisor.where(divisor.abs() > self.config.eps, np.nan)

        ratio = dividend / safe_divisor


        if clip is not None:

            ratio = ratio.clip(lower=-clip, upper=clip)


        df[output] = ratio



    def _safe_diff(self, df: pd.DataFrame, left: str, right: str, output: str) -> None:

        """
        Adds a difference feature.
        """

        if left not in df.columns or right not in df.columns:

            df[output] = np.nan

            return


        df[output] = df[left].astype(float) - df[right].astype(float)



    def _add_atr_context_features(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Adds ATR compression/expansion context features.
        
        atr_compression_ratio_N: current ATR / prior rolling mean ATR over N candles

        If < 1.0 = ATR, it is below recent average -> compression-like

        If > 1.0 = ATR it is above recent average -> expansion-like

        Using shift(1) avoids the current row influencing its own context.
        """

        atr_col = f"atr_{self.config.atr_window}"


        if atr_col not in df.columns:

            raise ValueError(f"Missing ATR column: {atr_col}. " "Call _add_atr() before _add_atr_context_features().")

        
        for window in sorted(self.config.windows):

            atr_mean_col = f"atr_mean_{window}"

            ratio_col = f"atr_compression_ratio_{window}"

            df[atr_mean_col] = (df[atr_col].rolling( window=window, min_periods=self._min_periods(window))
                                                                                                          .mean()
                                                                                                          .shift(1))

            safe_mean = df[atr_mean_col].where( df[atr_mean_col].abs() > self.config.eps, np.nan)

            
            df[ratio_col] = (df[atr_col] / safe_mean).clip(lower=0.0, upper=10.0)

        

        return df





    # ---------------------------------------------------------------------
    #                            Helper Functions
    # ---------------------------------------------------------------------