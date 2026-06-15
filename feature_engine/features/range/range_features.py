from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from pathlib import Path


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
        inside the upper/lower portion of the rolling range. I will respect this assumption even though this classifier is not meant to predict
        future behavior.
        
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

    windows: tuple[int, ...] = (20, 50)
    
    atr_window: int = 14

    atr_method: str = "wilder"
    
    zone_pct: float = 0.15
    
    slope_window: int = 5

    #Note that it is bigger than main window. I am asking "Is behavior in window unusual from previous data?
    zscore_windows: tuple[int, ...] = (100, 250) 

    zscore_clip: float = 5.0
    
    min_periods_ratio: float = 0.8

    #Thresholds set for handling abnormalities in a range
    quantile_low: float = 0.05
    quantile_high: float = 0.95

    #Improves decision-making consistency/stability
    persistence_thresholds: tuple[float, ...] = (0.6, 0.7)

    #abnormal_volume_zscore_threshold: float = 2.0 Unused for now

    #abnormal_range_zscore_threshold: float = 2.0 Unused for now

    time_context_timezone: str = "UTC"
    
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

        8. Slope / flatness

        9. Volatility compression

        10. Lifecycle/change features

        11. Multi-window comparison features
    """

    def __init__(
        self,
        config: Optional[RangeFeatureConfig] = None,
        *,
        windows: Optional[Iterable[int]] = None,
        atr_window: Optional[int] = None,
        atr_method: Optional[str] = None,
        zone_pct: Optional[float] = None,
        slope_window: Optional[int] = None,
        zscore_windows: Optional[Iterable[int]] = None,
        zscore_clip: Optional[float] = None) -> None:

        
        self.config = config or RangeFeatureConfig()

        if windows is not None:

            self.config.windows = tuple(windows)

        if atr_window is not None:

            self.config.atr_window = atr_window

        if atr_method is not None:

            self.config.atr_method = atr_method

        if zone_pct is not None:

            self.config.zone_pct = zone_pct

        if slope_window is not None:

            self.config.slope_window = slope_window

        if zscore_windows is not None:

            self.config.zscore_windows = tuple(zscore_windows)

        if zscore_clip is not None:
    
            self.config.zscore_clip = zscore_clip

        self.config.windows = tuple(sorted(self.config.windows))
        self.config.zscore_windows = tuple(sorted(self.config.zscore_windows))

        self._validate_config()

        
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Main feature extraction method.

        Inputs df: Normalized OHLC dataframe. In this case data from MarketNormalizationEngine.

        Returns pd.DataFrame: Original dataframe plus range-behavior features.
        """
    
        data = df.copy()

        self._validate_input_schema(data)
    
        self._validate_ohlc_data(data)
    
        data["timestamp"] = pd.to_datetime(data["timestamp"])
    
        data = data.sort_values("timestamp").reset_index(drop=True)
    
        # ------------------------------------------------------------
        # ATR
        # ------------------------------------------------------------
    
        atr_features = self._build_atr_features(data)
        data = pd.concat([data, atr_features], axis=1)

        # ------------------------------------------------------------
        # Candles
        # ------------------------------------------------------------
        candle_structure = self._build_candle_structure_features(data)
        data = pd.concat([data, candle_structure], axis=1)

        
        # ------------------------------------------------------------ 
        # per-window base features
        # ------------------------------------------------------------

        #Cannot concatenate everything at once since some features rely on others
        for window in self.config.windows:

            #Geometry features
            geometry = self._build_range_geometry_features(data, window)    
            data = pd.concat([data, geometry], axis=1)

            quantile_geometry = self._build_quantile_range_geometry_features(data, window)  
            data = pd.concat([data, quantile_geometry], axis=1)

            #Base features
            boundary = self._build_boundary_touch_features(data, window)
    
            rotation = self._build_rotation_features(data, window)
    
            directional = self._build_directional_efficiency_features(data, window)
    
            slope = self._build_slope_features(data, window)

            breakout = self._build_breakout_reentry_features(data, window)

            close_location = self._build_close_location_features(data, window)

            candle_body = self._build_body_range_features(data, window)

            
            base_features = pd.concat( [directional, boundary, rotation, slope, breakout, close_location, candle_body], axis=1)
            data = pd.concat([data, base_features], axis=1)

            
            #Lifecycle features
            lifecycle = self._build_lifecycle_features(data, window)
    
            data = pd.concat([data, lifecycle], axis=1)

        
        # ------------------------------------------------------------
        # cross-window context
        # ------------------------------------------------------------
    
        atr_context = self._build_atr_context_features(data)
        data = pd.concat([data, atr_context], axis=1)
        
        range_candidates = self._build_range_behavior_candidates(data)
        data = pd.concat([data, range_candidates], axis=1)

        persistence = self._build_persistence_features(data)
        data = pd.concat([data, persistence], axis=1)

        post_candidate_lifecycle = self._build_post_candidate_lifecycle_features(data)
        data = pd.concat([data, post_candidate_lifecycle], axis=1)
        
        acceleration = self._build_acceleration_features(data)
        data = pd.concat([data, acceleration], axis=1)
        
        multi_window = self._build_multi_window_comparison_features(data)
        data = pd.concat([data, multi_window], axis=1)

        volume_context = self._build_volume_context_features(data)
        data = pd.concat([data, volume_context], axis=1)

        time_context = self._build_time_context_features(data)
        data = pd.concat([data, time_context], axis=1)
        
        # ------------------------------------------------------------
        # rolling z-scores
        # ------------------------------------------------------------
    
        zscores = self._build_rolling_zscores(data)
        data = pd.concat([data, zscores], axis=1)


        
        data = data.copy()

        
        return data

    

    # ---------------------------------------------------------------------
    #                          Rolling Z-SCORE
    # ---------------------------------------------------------------------


    def _zscore_feature_columns(self, df: pd.DataFrame) -> list[str]:

        """
        Selects behavior features that benefit from rolling z-score.
        Avoids raw prices, timestamps, boolean zone flags, and already-zscored columns.
        """
    
        include_keywords = [
            "range_width_atr_",
            "quantile_range_width_atr_",
            "range_outlier_sensitivity_",
            "quantile_position_in_range_",
            "quantile_distance_from_mid_",
        
            "directional_efficiency_",
            "mid_cross_frequency_",
            "rotation_score_",
            "touch_balance_",
            "two_sided_touch_score_",
            "boundary_activity_score_",
        
            "abs_robust_trendline_move_atr_",
            "flatness_score_",
            "slope_outlier_sensitivity_",
        
            "failed_break_frequency_",
            "close_outside_frequency_",
            "upper_break_distance_atr_",
            "lower_break_distance_atr_",
        
            "avg_close_location_",
            "close_location_std_",
            "upper_close_frequency_",
            "lower_close_frequency_",
            "middle_close_frequency_",
            "close_location_imbalance_",
        
            "avg_body_to_range_ratio_",
            "body_to_range_ratio_std_",
            "large_body_frequency_",
            "small_body_frequency_",
            "avg_candle_range_atr_",
            "avg_body_size_atr_",
            "directional_body_pressure_",
            "bullish_candle_frequency_",
            "bearish_candle_frequency_",
            "avg_upper_wick_ratio_",
            "avg_lower_wick_ratio_",
            "avg_wick_imbalance_",
        
            "atr_compression_ratio_",
            "one_sided_position_pressure_",
            "range_behavior_candidate_",
            "range_candidate_persistence_",
            "compression_persistence_",
            "one_sided_pressure_persistence_",
        
            "range_agreement_",
            "range_component_agreement_",
            "range_candidate_agreement_",
        
            "volume_ratio_",
            "volume_zscore_",
            "volume_near_upper_share_",
            "volume_near_lower_share_",
            "volume_boundary_imbalance_",
        
            "_acceleration_"
        ]

        
        exclude_keywords = [
            "range_high_",
            "range_low_",
            "range_mid_",
            "quantile_range_high_",
            "quantile_range_low_",
            "quantile_range_mid_",
            "upper_zone_start_",
            "lower_zone_end_",
            "near_upper_zone_",
            "near_lower_zone_",
            "timestamp",
            "total_volume"
        ]
    
        
        selected = []
    
        for col in df.columns:

            
            if self._is_zscore_column(col):

                continue
    
            if any(excluded in col for excluded in exclude_keywords):
    
                continue
    
            
            if any(included in col for included in include_keywords):
    
                selected.append(col)
    
        
        return selected


    
    def _build_rolling_zscores(self, df: pd.DataFrame) -> pd.DataFrame:
    
        """
        Adds rolling z-scores.
    
        Z Score = ((X - u) / σ) where X is the most recent data point, u is the rolling average (mean) over the defined lookback window, and 
        sigma is the rolling standard deviation over the same window.
            
        The rolling mean/std are shifted by 1 so the current row is compared
        only against prior feature history. This avoids the current value 
        influencing its own normalization.
        """

        features: dict[str, pd.Series] = {}

        
        for base_col in self._zscore_feature_columns(df):

            
            if base_col not in df.columns:
    
                continue

            
            x = df[base_col].astype(float)
    
            for z_window in self.config.zscore_windows:
    
                min_periods = self._min_periods(z_window)
    
                rolling_mean = ( x.rolling(window=z_window, min_periods=min_periods).mean().shift(1))

                
                rolling_std = (x.rolling(window=z_window, min_periods=min_periods).std(ddof=0).shift(1))
                safe_std = rolling_std.where( rolling_std > self.config.eps, np.nan)

                
                z = (x - rolling_mean) / safe_std

                
                features[f"{base_col}_z{z_window}"] = z.clip(
                    lower=-self.config.zscore_clip,
                    upper=self.config.zscore_clip)

        
        return pd.DataFrame(features, index=df.index)


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

        if not 0.0 < self.config.quantile_low < 0.5:
            
            raise ValueError("quantile_low must be between 0.0 and 0.5.")

        if not 0.5 < self.config.quantile_high < 1.0:
        
            raise ValueError("quantile_high must be between 0.5 and 1.0.")

        if self.config.quantile_low >= self.config.quantile_high:
            
            raise ValueError("quantile_low must be less than quantile_high.")

        if any( threshold <= 0.0 or threshold >= 1.0 for threshold in self.config.persistence_thresholds):
    
            raise ValueError("persistence_thresholds must be between 0.0 and 1.0.")

    

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

        Uses NumPy vectorization instead of Python nested loops.
        Still O(n^2).
        
        TODO: Upgrade to a multithreaded implementation.
        """

        values = np.asarray(values, dtype=float)

        
        if len(values) < 2:

            return np.nan

        if np.isnan(values).any():

            return np.nan

        n = len(values)

        idx_i, idx_j = np.triu_indices(n, k=1)

        slopes = (values[idx_j] - values[idx_i]) / (idx_j - idx_i)

        
        return float(np.median(slopes))


    
    @staticmethod
    def _linear_regression_slope(values: np.ndarray) -> float:

        values = np.asarray(values, dtype=float)
    
        if len(values) < 2:
    
            return np.nan
    
        if np.isnan(values).any():
    
            return np.nan
    
        x = np.arange(len(values), dtype=float)
    
        x_mean = x.mean()
    
        y_mean = values.mean()
    
        denominator = np.sum((x - x_mean) ** 2)
    
        if denominator == 0:
    
            return 0.0
    
        numerator = np.sum((x - x_mean) * (values - y_mean))
    
        return float(numerator / denominator)

    

    def _build_slope_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        
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

        close_slope = (df["close"].rolling(window=window, min_periods=min_periods).apply(self._linear_regression_slope, raw=True))

        close_slope_atr = close_slope / (df[atr_col] + c.eps)

        abs_close_slope_atr = close_slope_atr.abs()

        trendline_move_atr = ( close_slope * (window - 1)) / (df[atr_col] + c.eps)

        abs_trendline_move_atr = trendline_move_atr.abs()

        robust_close_slope = (df["close"].rolling(window=window, min_periods=min_periods).apply(self._theil_sen_slope, raw=True))

        robust_close_slope_atr = robust_close_slope / (df[atr_col] + c.eps)

        abs_robust_close_slope_atr = robust_close_slope_atr.abs()

        robust_trendline_move_atr = (robust_close_slope * (window - 1)) / (df[atr_col] + c.eps)

        abs_robust_trendline_move_atr = robust_trendline_move_atr.abs()

        flatness_score = (1.0 / (1.0 + abs_robust_trendline_move_atr)).clip(lower=0.0, upper=1.0)

        slope_outlier_sensitivity = (abs_trendline_move_atr - abs_robust_trendline_move_atr).abs()
        

        return pd.DataFrame(

            {
                f"close_slope_{window}": close_slope,
                f"close_slope_atr_{window}": close_slope_atr,
                f"abs_close_slope_atr_{window}": abs_close_slope_atr,
                f"trendline_move_atr_{window}": trendline_move_atr,
                f"abs_trendline_move_atr_{window}": abs_trendline_move_atr,
                f"robust_close_slope_{window}": robust_close_slope,
                f"robust_close_slope_atr_{window}": robust_close_slope_atr,
                f"abs_robust_close_slope_atr_{window}": abs_robust_close_slope_atr,
                f"robust_trendline_move_atr_{window}": robust_trendline_move_atr,
                f"abs_robust_trendline_move_atr_{window}": abs_robust_trendline_move_atr,
                f"flatness_score_{window}": flatness_score,
                f"slope_outlier_sensitivity_{window}": slope_outlier_sensitivity,
            }, index=df.index)



    # ---------------------------------------------------------------------
    #                                  ATR
    # ---------------------------------------------------------------------
    def _build_atr_features(self, df: pd.DataFrame) -> pd.DataFrame:

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

        
        true_range = true_range_components.max(axis=1)

        
        atr = true_range.ewm(alpha=1.0 / c.atr_window, adjust=False, min_periods=self._min_periods(c.atr_window)).mean()

        
        return pd.DataFrame({"true_range": true_range, atr_col: atr}, index=df.index)



    # ---------------------------------------------------------------------
    #                               Geometry
    # ---------------------------------------------------------------------
    
    def _build_range_geometry_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:

        c = self.config
        min_periods = self._min_periods(window)

        
        atr_col = f"atr_{c.atr_window}"

        range_high = df["high"].rolling(window, min_periods=min_periods).max()
        range_low = df["low"].rolling(window, min_periods=min_periods).min()
        range_mid = (range_high + range_low) / 2.0
        range_width = range_high - range_low
    
        upper_zone_start = range_high - (range_width * c.zone_pct)
        lower_zone_end = range_low + (range_width * c.zone_pct)
    
        position_in_range = (( df["close"] - range_low) / (range_width + c.eps )).clip(lower=0.0, upper=1.0)
    
        distance_to_range_high = range_high - df["close"]
        distance_to_range_low = df["close"] - range_low


        return pd.DataFrame(
        {
            f"range_high_{window}": range_high,
            f"range_low_{window}": range_low,
            f"range_mid_{window}": range_mid,
            f"range_width_{window}": range_width,
            f"range_width_atr_{window}": range_width / (df[atr_col] + c.eps),
            f"upper_zone_start_{window}": upper_zone_start,
            f"lower_zone_end_{window}": lower_zone_end,
            f"position_in_range_{window}": position_in_range,
            f"distance_to_range_high_{window}": distance_to_range_high,
            f"distance_to_range_low_{window}": distance_to_range_low,
            f"distance_to_range_high_atr_{window}": distance_to_range_high / (df[atr_col] + c.eps),
            f"distance_to_range_low_atr_{window}": distance_to_range_low / (df[atr_col] + c.eps),
            f"current_distance_from_mid_{window}": ((df["close"] - range_mid).abs() / (range_width + c.eps))
        }, index=df.index)


    
    def _build_quantile_range_geometry_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:

        """
        Adds quantile-based range geometry.

        I want to help the rolling max/min range from being distorted by one abnormal wick.
        Using rolling quantiles to estimate the core range could possibly help.

        Features:

            quantile_range_high_N
            quantile_range_low_N
            quantile_range_mid_N
            quantile_range_width_N
            quantile_range_width_atr_N
            range_outlier_sensitivity_N
            quantile_position_in_range_N
            quantile_distance_to_high_atr_N
            quantile_distance_to_low_atr_N

        """

        c = self.config

        min_periods = self._min_periods(window)

        atr_col = f"atr_{c.atr_window}"

        
        quantile_high = ( df["high"].rolling(window, min_periods=min_periods).quantile(c.quantile_high))
        quantile_low = ( df["low"].rolling(window, min_periods=min_periods).quantile(c.quantile_low))

        
        quantile_mid = (quantile_high + quantile_low) / 2.0
        quantile_width = quantile_high - quantile_low
        quantile_width_atr = quantile_width / (df[atr_col] + c.eps)

        standard_width_atr_col = f"range_width_atr_{window}"

        if standard_width_atr_col in df.columns:

            range_outlier_sensitivity = ( df[standard_width_atr_col] - quantile_width_atr).clip(lower=0.0)

        
        else:

            range_outlier_sensitivity = pd.Series(np.nan, index=df.index)

        
        quantile_position = ( (df["close"] - quantile_low) / (quantile_width + c.eps)).clip(lower=0.0, upper=1.0)

        
        quantile_distance_to_high = quantile_high - df["close"]
        quantile_distance_to_low = df["close"] - quantile_low

        
        return pd.DataFrame(
            {
                f"quantile_range_high_{window}": quantile_high,
                f"quantile_range_low_{window}": quantile_low,
                f"quantile_range_mid_{window}": quantile_mid,
                f"quantile_range_width_{window}": quantile_width,
                f"quantile_range_width_atr_{window}": quantile_width_atr,
                f"range_outlier_sensitivity_{window}": range_outlier_sensitivity,
                f"quantile_position_in_range_{window}": quantile_position,
                f"quantile_distance_to_high_atr_{window}": ( quantile_distance_to_high / (df[atr_col] + c.eps)),
                f"quantile_distance_to_low_atr_{window}": ( quantile_distance_to_low / (df[atr_col] + c.eps)),
                f"quantile_distance_from_mid_{window}": ( (df["close"] - quantile_mid).abs() / (quantile_width + c.eps)),
            }, index=df.index)

        
    # ---------------------------------------------------------------------
    #                              Directional
    # ---------------------------------------------------------------------
    def _build_directional_efficiency_features(self,df: pd.DataFrame, window: int) -> pd.DataFrame:

        c = self.config

        min_periods = self._min_periods(window)

        net_change = (df["close"] - df["close"].shift(window - 1)).abs()

        total_movement = (df["close"].diff().abs().rolling(window, min_periods=min_periods).sum())

        return pd.DataFrame({ f"directional_efficiency_{window}": net_change / (total_movement + c.eps) },index=df.index)


    # ---------------------------------------------------------------------
    #                              Rotational
    # ---------------------------------------------------------------------

    def _build_rotation_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:

        c = self.config

        min_periods = self._min_periods(window)
        mid_col = f"range_mid_{window}"
        width_col = f"range_width_{window}"
    
        above_mid = (df["close"] > df[mid_col]).where( df[mid_col].notna(), np.nan)
    
        mid_cross = (above_mid != above_mid.shift(1)).astype(float)
        
        mid_cross = mid_cross.where( above_mid.notna() & above_mid.shift(1).notna(), np.nan)
    
        mid_cross_count = mid_cross.rolling( window, min_periods=min_periods).sum()
    
        mid_cross_frequency = mid_cross_count / window
    
        avg_distance_from_mid = ( ((df["close"] - df[mid_col]).abs() / (df[width_col] + c.eps)).rolling(window, min_periods=min_periods).mean())
    
        rotation_score = (mid_cross_frequency * (1.0 - avg_distance_from_mid.clip(0.0, 1.0))).clip(lower=0.0, upper=1.0)
    
    
        
        return pd.DataFrame(
            {
                f"mid_cross_count_{window}": mid_cross_count,
                f"mid_cross_frequency_{window}": mid_cross_frequency,
                f"avg_distance_from_mid_{window}": avg_distance_from_mid,
                f"rotation_score_{window}": rotation_score,
            }, index=df.index)



    
    def _build_close_location_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:

        """
        Describes where candles close inside their own high-low range and inside the rolling market range.
        """

        c = self.config

        min_periods = self._min_periods(window)

        candle_range = df["high"] - df["low"]

        close_location_value = ((df["close"] - df["low"]) / (candle_range + c.eps)).clip(lower=0.0, upper=1.0)

        
        upper_close = (close_location_value >= 1.0 - c.zone_pct).astype(float)
        lower_close = (close_location_value <= c.zone_pct).astype(float)
        middle_close = ((close_location_value > c.zone_pct)& (close_location_value < 1.0 - c.zone_pct)).astype(float)

        avg_close_location = close_location_value.rolling(window,min_periods=min_periods).mean()

        close_location_std = close_location_value.rolling(window, min_periods=min_periods).std(ddof=0)

        upper_close_frequency = upper_close.rolling(window, min_periods=min_periods).mean()
        lower_close_frequency = lower_close.rolling(window, min_periods=min_periods).mean()
        middle_close_frequency = middle_close.rolling(window, min_periods=min_periods).mean()

        close_location_imbalance = (upper_close_frequency - lower_close_frequency)

        return pd.DataFrame(
            {
                f"close_location_value_{window}": close_location_value,
                f"avg_close_location_{window}": avg_close_location,
                f"close_location_std_{window}": close_location_std,
                f"upper_close_frequency_{window}": upper_close_frequency,
                f"lower_close_frequency_{window}": lower_close_frequency,
                f"middle_close_frequency_{window}": middle_close_frequency,
                f"close_location_imbalance_{window}": close_location_imbalance,

            }, index=df.index)

    def _build_candle_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        """
        Adds candle-level body/range structure to describe the current candle itself.
        They are not rolling-window features, so they are only created once.
        """
    
        c = self.config
        atr_col = f"atr_{c.atr_window}"
    
        candle_range = df["high"] - df["low"]
        body_size = (df["close"] - df["open"]).abs()
    
        body_to_range_ratio = ( body_size / (candle_range + c.eps)).clip(lower=0.0, upper=1.0)
    
        candle_range_atr = candle_range / (df[atr_col] + c.eps)
        body_size_atr = body_size / (df[atr_col] + c.eps)
    
        candle_direction = np.sign(df["close"] - df["open"])
    
        upper_wick = df["high"] - pd.concat([df["open"], df["close"]], axis=1).max(axis=1)
        lower_wick = pd.concat([df["open"], df["close"]], axis=1).min(axis=1) - df["low"]
    
        upper_wick_ratio = (upper_wick / (candle_range + c.eps) ).clip(lower=0.0, upper=1.0)
        lower_wick_ratio = (lower_wick / (candle_range + c.eps)).clip(lower=0.0, upper=1.0)

        
        wick_imbalance = upper_wick_ratio - lower_wick_ratio

        
        return pd.DataFrame(
            {
                "candle_range": candle_range,
                "body_size": body_size,
                "body_to_range_ratio": body_to_range_ratio,
                "candle_range_atr": candle_range_atr,
                "body_size_atr": body_size_atr,
                "candle_direction": candle_direction,
                "upper_wick_ratio": upper_wick_ratio,
                "lower_wick_ratio": lower_wick_ratio,
                "wick_imbalance": wick_imbalance,
            }, index=df.index)



    def _build_body_range_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        
        """
        Adds rolling candle body/range behavior.
    
        These features summarize candle structure over the current rolling window.
        The raw candle-level features are created separately in
        _build_candle_structure_features().
        """
    
        min_periods = self._min_periods(window)
    
        required_cols = [
            "body_to_range_ratio",
            "candle_range_atr",
            "body_size_atr",
            "candle_direction",
            "upper_wick_ratio",
            "lower_wick_ratio",
            "wick_imbalance"]

        
        missing = [col for col in required_cols if col not in df.columns]

        
        if missing:
            
            raise ValueError(f"Missing candle structure columns before body/range rolling features: {missing}. "
                "Call _build_candle_structure_features() first."
            )

        
        body_to_range_ratio = df["body_to_range_ratio"]
        candle_range_atr = df["candle_range_atr"]
        body_size_atr = df["body_size_atr"]
        candle_direction = df["candle_direction"]
        upper_wick_ratio = df["upper_wick_ratio"]
        lower_wick_ratio = df["lower_wick_ratio"]
        wick_imbalance = df["wick_imbalance"]
    
        large_body = (body_to_range_ratio >= 0.7).astype(float)
        small_body = (body_to_range_ratio <= 0.3).astype(float)
    
        bullish_candle = (candle_direction > 0).astype(float)
        bearish_candle = (candle_direction < 0).astype(float)
    
        avg_body_to_range = body_to_range_ratio.rolling( window, min_periods=min_periods).mean()
    
        body_to_range_std = body_to_range_ratio.rolling(window, min_periods=min_periods).std(ddof=0)
    
        large_body_frequency = large_body.rolling(window,min_periods=min_periods).mean()
    
        small_body_frequency = small_body.rolling(window, min_periods=min_periods).mean()
    
        avg_candle_range_atr = candle_range_atr.rolling(window, min_periods=min_periods).mean()
    
        avg_body_size_atr = body_size_atr.rolling( window, min_periods=min_periods).mean()
    
        directional_body_pressure = (candle_direction * body_to_range_ratio).rolling(window, min_periods=min_periods).mean()
    
        bullish_candle_frequency = bullish_candle.rolling(window, min_periods=min_periods).mean()
    
        bearish_candle_frequency = bearish_candle.rolling(window, min_periods=min_periods).mean()
    
        avg_upper_wick_ratio = upper_wick_ratio.rolling(window,min_periods=min_periods).mean()
    
        avg_lower_wick_ratio = lower_wick_ratio.rolling(window, min_periods=min_periods).mean()
    
        avg_wick_imbalance = wick_imbalance.rolling(window, min_periods=min_periods).mean()
    
        return pd.DataFrame(
            {
                f"avg_body_to_range_ratio_{window}": avg_body_to_range,
                f"body_to_range_ratio_std_{window}": body_to_range_std,
                f"large_body_frequency_{window}": large_body_frequency,
                f"small_body_frequency_{window}": small_body_frequency,
                f"avg_candle_range_atr_{window}": avg_candle_range_atr,
                f"avg_body_size_atr_{window}": avg_body_size_atr,
                f"directional_body_pressure_{window}": directional_body_pressure,
                f"bullish_candle_frequency_{window}": bullish_candle_frequency,
                f"bearish_candle_frequency_{window}": bearish_candle_frequency,
                f"avg_upper_wick_ratio_{window}": avg_upper_wick_ratio,
                f"avg_lower_wick_ratio_{window}": avg_lower_wick_ratio,
                f"avg_wick_imbalance_{window}": avg_wick_imbalance,
            }, index=df.index)
        
        
    # ---------------------------------------------------------------------
    #                              Lifecycle
    # ---------------------------------------------------------------------
    def _build_lifecycle_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:

        """
        Adds current-change style features. 
        These help describe lifecycle:

            - forming
            - stabilizing
            - compressing
            - expanding
            - weakening
            - transitioning
        """

        c = self.config

        sw = c.slope_window

        features: dict[str, pd.Series] = {}

        
        lifecycle_base_cols = [
            f"range_width_atr_{window}",
            f"quantile_range_width_atr_{window}",
            f"range_outlier_sensitivity_{window}",
            f"directional_efficiency_{window}",
            f"touch_balance_{window}",
            f"mid_cross_frequency_{window}",
            f"boundary_activity_score_{window}",
            f"two_sided_touch_score_{window}",
            f"rotation_score_{window}",
            f"flatness_score_{window}",
            f"position_in_range_{window}",
            f"failed_break_frequency_{window}",
            f"close_outside_frequency_{window}",
            f"avg_close_location_{window}",
            f"close_location_imbalance_{window}",
            f"avg_body_to_range_ratio_{window}",
            f"large_body_frequency_{window}",
            f"small_body_frequency_{window}",
            f"directional_body_pressure_{window}",
            f"bullish_candle_frequency_{window}",
            f"bearish_candle_frequency_{window}",
            f"avg_upper_wick_ratio_{window}",
            f"avg_lower_wick_ratio_{window}",
            f"avg_wick_imbalance_{window}"
        ]
        

        for col in lifecycle_base_cols:

            if col not in df.columns:

                continue

            
            features[f"{col}_change_{sw}"] = df[col] - df[col].shift(sw)

            features[f"{col}_slope_{sw}"] = (df[col].rolling(sw, min_periods=max(2, int(sw * 0.8))).apply(self._linear_regression_slope, raw=True))

        
        width_col = f"range_width_atr_{window}"

        width_slope_col = f"{width_col}_slope_{sw}"

        if width_slope_col in features:

            features[f"range_expansion_pressure_{window}"] = (features[width_slope_col].clip(lower=0.0))

            
            features[f"range_compression_pressure_{window}"] = ((-features[width_slope_col]).clip(lower=0.0))

        
        else:

            features[f"range_expansion_pressure_{window}"] = pd.Series(np.nan, index=df.index)

            features[f"range_compression_pressure_{window}"] = pd.Series(np.nan, index=df.index)

        
        de_slope_col = f"directional_efficiency_{window}_slope_{sw}"

        features[f"directional_pressure_change_{window}"] = features.get(de_slope_col, pd.Series(np.nan, index=df.index))

        pos_col = f"position_in_range_{window}"

        
        if pos_col in df.columns:

            time_near_upper = ((df[pos_col] >= 1.0 - c.zone_pct).astype(float).rolling(window, min_periods=self._min_periods(window)).mean())

            time_near_lower = ((df[pos_col] <= c.zone_pct).astype(float).rolling(window, min_periods=self._min_periods(window)).mean())

            
            one_sided_position_pressure = pd.concat([time_near_upper, time_near_lower], axis=1).max(axis=1)

        
        else:

            time_near_upper = pd.Series(np.nan, index=df.index)

            time_near_lower = pd.Series(np.nan, index=df.index)

            one_sided_position_pressure = pd.Series(np.nan, index=df.index)

        
        features[f"time_near_upper_{window}"] = time_near_upper

        features[f"time_near_lower_{window}"] = time_near_lower

        features[f"one_sided_position_pressure_{window}"] = one_sided_position_pressure

        return pd.DataFrame(features, index=df.index)


    def _build_acceleration_features(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Adds second-order change features.
        If slope tells us a behavior is changing, acceleration tells us whether that change is speeding up or slowing down.
        """

        sw = self.config.slope_window

        features: dict[str, pd.Series] = {}

        acceleration_base_names = [
            "range_width_atr",
            "quantile_range_width_atr",
            "range_outlier_sensitivity",
            "directional_efficiency",
            "touch_balance",
            "mid_cross_frequency",
            "boundary_activity_score",
            "two_sided_touch_score",
            "rotation_score",
            "flatness_score",
            "position_in_range",
            "failed_break_frequency",
            "close_outside_frequency",
            "avg_close_location",
            "close_location_imbalance",
            "avg_body_to_range_ratio",
            "large_body_frequency",
            "small_body_frequency",
            "directional_body_pressure",
            "bullish_candle_frequency",
            "bearish_candle_frequency",
            "avg_upper_wick_ratio",
            "avg_lower_wick_ratio",
            "avg_wick_imbalance"
        ]


        
        for window in sorted(self.config.windows):

            for base_name in acceleration_base_names:

                slope_col = f"{base_name}_{window}_slope_{sw}"

                if slope_col not in df.columns:

                    continue

                acceleration_col = f"{base_name}_{window}_acceleration_{sw}"

                features[acceleration_col] = df[slope_col] - df[slope_col].shift(sw)

        
        return pd.DataFrame(features, index=df.index)

        
    # ---------------------------------------------------------------------
    #                              Boundaries
    # ---------------------------------------------------------------------
    def _build_boundary_touch_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:

        c = self.config

        min_periods = self._min_periods(window)

        upper_zone_col = f"upper_zone_start_{window}"
        lower_zone_col = f"lower_zone_end_{window}"
        near_upper = (df["high"] >= df[upper_zone_col]).astype(float)
        near_lower = (df["low"] <= df[lower_zone_col]).astype(float)
        upper_touch_count = near_upper.rolling(window, min_periods=min_periods).sum()

        lower_touch_count = near_lower.rolling(window, min_periods=min_periods).sum()

        total_touch_count = upper_touch_count + lower_touch_count

        boundary_activity_score = (total_touch_count / (2.0 * window)).clip(lower=0.0, upper=1.0)

        max_touches = pd.concat([upper_touch_count, lower_touch_count], axis=1).max(axis=1)

        min_touches = pd.concat([upper_touch_count, lower_touch_count], axis=1).min(axis=1)

        touch_balance = min_touches / (max_touches + c.eps)

        two_sided_touch_score = (touch_balance * boundary_activity_score).clip(lower=0.0, upper=1.0)

        
        return pd.DataFrame(
            {
                f"near_upper_zone_{window}": near_upper,
                f"near_lower_zone_{window}": near_lower,
                f"upper_touch_count_{window}": upper_touch_count,
                f"lower_touch_count_{window}": lower_touch_count,
                f"upper_touch_frequency_{window}": upper_touch_count / window,
                f"lower_touch_frequency_{window}": lower_touch_count / window,
                f"total_touch_count_{window}": total_touch_count,
                f"boundary_activity_score_{window}": boundary_activity_score,
                f"touch_balance_{window}": touch_balance,
                f"two_sided_touch_score_{window}": two_sided_touch_score,
            }, index=df.index)


    def _build_breakout_reentry_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:

        """
        Describes range violations and failed breaks.
        Attempts to describe whether the current candle interacted with or violated the rolling range boundaries.

        Uses prior range boundaries via shift(1) so the current candle does not define the boundary it is being tested against.

        The goal is to account for clean range, range with sweeps, failed breakout, actual boundary violation, and transition behavior.
        """

        c = self.config

        atr_col = f"atr_{c.atr_window}"

        high_boundary = df[f"range_high_{window}"].shift(1)
        low_boundary = df[f"range_low_{window}"].shift(1)

        if f"quantile_range_high_{window}" in df.columns:
            
            quantile_high_boundary = df[f"quantile_range_high_{window}"].shift(1)
            
        else:
            
            quantile_high_boundary = high_boundary
        
        if f"quantile_range_low_{window}" in df.columns:
            
            quantile_low_boundary = df[f"quantile_range_low_{window}"].shift(1)
        
        else:
            
            quantile_low_boundary = low_boundary

        high_break = (df["high"] > high_boundary).astype(float)
        low_break = (df["low"] < low_boundary).astype(float)

        close_above = (df["close"] > high_boundary).astype(float)
        close_below = (df["close"] < low_boundary).astype(float)

        failed_break_upper = ((df["high"] > high_boundary) & (df["close"] <= high_boundary)).astype(float)
        failed_break_lower = ((df["low"] < low_boundary) & (df["close"] >= low_boundary)).astype(float)

        quantile_high_break = (df["high"] > quantile_high_boundary).astype(float)
        quantile_low_break = (df["low"] < quantile_low_boundary).astype(float)

        upper_break_distance_atr = ( (df["high"] - high_boundary).clip(lower=0.0)/ (df[atr_col] + c.eps))  
        lower_break_distance_atr = ((low_boundary - df["low"]).clip(lower=0.0) / (df[atr_col] + c.eps))

        close_outside_range = ( (df["close"] > high_boundary) | (df["close"] < low_boundary)).astype(float)

        wick_break_no_close_outside = (((df["high"] > high_boundary) | (df["low"] < low_boundary)) & (close_outside_range == 0)).astype(float)

        min_periods = self._min_periods(window)

        failed_break_frequency = ((failed_break_upper + failed_break_lower).rolling(window, min_periods=min_periods).sum()/ window)

        
        close_outside_frequency = (close_outside_range.rolling(window, min_periods=min_periods).sum() / window)

        
        return pd.DataFrame(
            
            {
                f"high_break_above_range_{window}": high_break,
                f"low_break_below_range_{window}": low_break,
                f"close_above_range_{window}": close_above,
                f"close_below_range_{window}": close_below,
                f"failed_break_upper_{window}": failed_break_upper,
                f"failed_break_lower_{window}": failed_break_lower,
                f"quantile_high_break_{window}": quantile_high_break,
                f"quantile_low_break_{window}": quantile_low_break,
                f"upper_break_distance_atr_{window}": upper_break_distance_atr,
                f"lower_break_distance_atr_{window}": lower_break_distance_atr,
                f"close_outside_range_{window}": close_outside_range,
                f"wick_break_no_close_outside_{window}": wick_break_no_close_outside,
                f"failed_break_frequency_{window}": failed_break_frequency,
                f"close_outside_frequency_{window}": close_outside_frequency,
            
            }, index=df.index)



    def _build_time_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        """
        Adds calendar and trading-session context.
    
        For daily data:
            Calendar features are useful.
            Session features might be not as meaningful.
    
        For intraday data:
            Session features help describe market behavior differences across
            Asian, London, New York, overlap, and rollover periods.
    
        Assumes timestamps are UTC (this will change once integrating in broker)
        """
        
        timezone = self.config.time_context_timezone
        
        timestamp = pd.to_datetime(df["timestamp"])
    
        if timestamp.dt.tz is None:
            
            timestamp = timestamp.dt.tz_localize(timezone)
            
        else:
            timestamp = timestamp.dt.tz_convert(timezone)
    
        time_hour = timestamp.dt.hour
        day_of_week = timestamp.dt.dayofweek
        month = timestamp.dt.month
    
        # ------------------------------------------------------------
        # Calendar context
        # ------------------------------------------------------------
    
        is_monday = (day_of_week == 0).astype(float)
        is_friday = (day_of_week == 4).astype(float)
        is_weekend = (day_of_week >= 5).astype(float)
    
        is_month_start = timestamp.dt.is_month_start.astype(float)
        is_month_end = timestamp.dt.is_month_end.astype(float)
    
        is_new_year = ( (timestamp.dt.month == 1) & (timestamp.dt.day == 1)).astype(float)
    
        is_christmas = ((timestamp.dt.month == 12) & (timestamp.dt.day == 25)).astype(float)
    
        # ------------------------------------------------------------
        # forex session time approximations
        # ------------------------------------------------------------
        # approximate and intentionally simple.
    
    
        is_asian_session = ( (time_hour >= 0) & (time_hour < 7)).astype(float)
    
        is_london_session = ( (time_hour >= 7) & (time_hour < 16)).astype(float)
        is_new_york_session = ( (time_hour >= 12) & (time_hour < 21) ).astype(float)
        is_london_ny_overlap = ( (time_hour >= 12) & (time_hour < 16)).astype(float)
    
        
        #21:00–23:00 can be weird 
        is_rollover_window = ((time_hour >= 21) & (time_hour < 23) ).astype(float)
    
        
        is_asian_london_transition = ( (time_hour >= 6) & (time_hour < 8)).astype(float)
        is_london_ny_transition = ( (time_hour >= 11) & (time_hour < 13)).astype(float)
    
        
        #cyclical time encoding 
        hour_sin = np.sin(2.0 * np.pi * time_hour / 24.0)
        hour_cos = np.cos(2.0 * np.pi * time_hour / 24.0)
    
        day_sin = np.sin(2.0 * np.pi * day_of_week / 7.0)
        day_cos = np.cos(2.0 * np.pi * day_of_week / 7.0)
    
        
        return pd.DataFrame(
            {
                "time_hour": time_hour.astype(float),
                "day_of_week": day_of_week.astype(float),
                "month": month.astype(float),
    
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
                "day_sin": day_sin,
                "day_cos": day_cos,
    
                "is_monday": is_monday,
                "is_friday": is_friday,
                "is_weekend": is_weekend,
                "is_month_start": is_month_start,
                "is_month_end": is_month_end,
                "is_new_year": is_new_year,
                "is_christmas": is_christmas,
    
                "is_asian_session": is_asian_session,
                "is_london_session": is_london_session,
                "is_new_york_session": is_new_york_session,
                "is_london_ny_overlap": is_london_ny_overlap,
                "is_rollover_window": is_rollover_window,
                "is_asian_london_transition": is_asian_london_transition,
                "is_london_ny_transition": is_london_ny_transition}, index=df.index)

    # ---------------------------------------------------------------------
    #                           Window Comaparison
    # ---------------------------------------------------------------------
    
    def _make_safe_ratio(self, df: pd.DataFrame, numerator: str,  denominator: str, *, clip: float | None = 10.0) -> pd.Series:

        """
        Adds a safe ratio feature.

        Why? Raw ratios can explode when the denominator is ~ zero so I want to avoid creating misleading outliers.
        Try range_width_atr_20 / range_width_atr_50 as an example to understand this more.
        """

        
        if numerator not in df.columns or denominator not in df.columns:
    
            return pd.Series(np.nan, index=df.index)

        
        numer = df[numerator].astype(float)
        denom = df[denominator].astype(float)
    
        safe_denom = denom.where( denom.abs() > self.config.eps, np.nan)

        
        ratio = numer / safe_denom

        
        if clip is not None:
    
            ratio = ratio.clip(lower=-clip, upper=clip)

        
        return ratio


    def _make_safe_diff(self, df: pd.DataFrame, left: str, right: str) -> pd.Series:

        """
        Adds a difference feature.
        """

        if left not in df.columns or right not in df.columns:

            return pd.Series(np.nan, index=df.index)

        
        return df[left].astype(float) - df[right].astype(float)


        
    def _build_atr_context_features(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Adds ATR compression/expansion context features.
        
        atr_compression_ratio_N: current ATR / prior rolling mean ATR over N candles

        If < 1.0 = ATR, it is below recent average -> compression-like

        If > 1.0 = ATR it is above recent average -> expansion-like

        Using shift(1) avoids the current row influencing its own context.
        """

        atr_col = f"atr_{self.config.atr_window}"

        
        if atr_col not in df.columns:
    
            raise ValueError(f"Missing ATR column: {atr_col}. "
                              "Call _build_atr_features() before _build_atr_context_features().")


        
        features: dict[str, pd.Series] = {}

        
        for window in sorted(self.config.windows):

            
            atr_mean = (df[atr_col].rolling(window=window, min_periods=self._min_periods(window)).mean().shift(1))
    
            safe_mean = atr_mean.where(atr_mean.abs() > self.config.eps, np.nan)

            
            atr_compression_ratio = ( df[atr_col] / safe_mean).clip(lower=0.0, upper=10.0)

            
            features[f"atr_mean_{window}"] = atr_mean
            features[f"atr_compression_ratio_{window}"] = atr_compression_ratio

        
        return pd.DataFrame(features, index=df.index)




    def _build_range_behavior_candidates(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Creates a soft range-behavior candidate score for each configured window. This is NOT the final model probability.
        This is a rule based descriptive feature that helps summarize whether each window currently behaves like a range.

        High values mean low directional efficiency, flatness, decent midpoint rotation, two-sided boundary interaction or
        repeated boundary activity.

        This score helps the model because it compresses several useful range traits into one interpretable signal.

        Each signal gets weighted because not all signals are equally important.
        
        For example... inefficiency and flatness are strongest for range-vs-trend separation. Rotation and two sided touches 
        confirm balanced/ranging behavior. Also boundary activity helps, but should not dominate.
        """

        features: dict[str, pd.Series] = {}

        
        for window in sorted(self.config.windows):
    
            required_cols = [
                f"directional_efficiency_{window}",
                f"flatness_score_{window}",
                f"rotation_score_{window}",
                f"two_sided_touch_score_{window}",
                f"boundary_activity_score_{window}",
            ]
    
            
            candidate_col = f"range_behavior_candidate_{window}"
    
            if any(col not in df.columns for col in required_cols):
    
                features[candidate_col] = pd.Series(np.nan, index=df.index)
    
                continue
    
            inefficiency = 1.0 - df[f"directional_efficiency_{window}"].clip(0.0, 1.0)
    
            flatness = df[f"flatness_score_{window}"].clip(0.0, 1.0)
    
            rotation = df[f"rotation_score_{window}"].clip(0.0, 1.0)
    
            two_sided = df[f"two_sided_touch_score_{window}"].clip(0.0, 1.0)
    
            boundary_activity = df[f"boundary_activity_score_{window}"].clip(0.0, 1.0)
    
            features[candidate_col] = (
                0.30 * inefficiency
                + 0.25 * flatness
                + 0.20 * rotation
                + 0.15 * two_sided
                + 0.10 * boundary_activity
            ).clip(0.0, 1.0)

        
        return pd.DataFrame(features, index=df.index)


    def _build_persistence_features(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Adds persistence/time-in-state descriptors.
        Helps distinguish one candle that looks range-like from a sustained range-like regime.
        """

        features: dict[str, pd.Series] = {}

        for window in sorted(self.config.windows):

            min_periods = self._min_periods(window)

            candidate_col = f"range_behavior_candidate_{window}"
            compression_col = f"range_compression_pressure_{window}"
            one_sided_col = f"one_sided_position_pressure_{window}"

            if candidate_col in df.columns:

                
                for threshold in self.config.persistence_thresholds:

                    threshold_name = str(threshold).replace(".", "_")

                    features[f"range_candidate_above_{threshold_name}_frequency_{window}"] = (
                        (df[candidate_col] >= threshold)
                        .astype(float)
                        .rolling(window, min_periods=min_periods)
                        .mean())

                features[f"range_candidate_persistence_{window}"] = (
                    df[candidate_col]
                    .rolling(window, min_periods=min_periods)
                    .mean())

            else:

                features[f"range_candidate_persistence_{window}"] = pd.Series(np.nan, index=df.index)

            
            if compression_col in df.columns:

                features[f"compression_persistence_{window}"] = (
                    (df[compression_col] > 0)
                    .astype(float)
                    .rolling(window, min_periods=min_periods)
                    .mean())

            else:

                features[f"compression_persistence_{window}"] = pd.Series(np.nan, index=df.index)


            
            if one_sided_col in df.columns:

                features[f"one_sided_pressure_persistence_{window}"] = (
                    df[one_sided_col]
                    .rolling(window, min_periods=min_periods)
                    .mean())

            else:

                features[f"one_sided_pressure_persistence_{window}"] = pd.Series(
                    np.nan,
                    index=df.index)

        
        return pd.DataFrame(features, index=df.index)


    def _build_post_candidate_lifecycle_features(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Adds lifecycle/change/acceleration features for post-base context columns.
        These columns get created after the normal per window lifecycle step, so they need their own lifecycle method.
    
        Examples include:
            range_behavior_candidate_N
            atr_compression_ratio_N
            range_candidate_persistence_N
            compression_persistence_N
            one_sided_pressure_persistence_N
        """
    
        sw = self.config.slope_window
    
        features: dict[str, pd.Series] = {}
    
        
        base_names = ["range_behavior_candidate",
                      "atr_compression_ratio",
                      "range_candidate_persistence",
                      "compression_persistence",
                      "one_sided_pressure_persistence"
                     ]
    
        
        for window in sorted(self.config.windows):
    
            for base_name in base_names:
    
                col = f"{base_name}_{window}"
    
                if col not in df.columns:
    
                    continue
    
                change_col = f"{col}_change_{sw}"
                slope_col = f"{col}_slope_{sw}"
                acceleration_col = f"{col}_acceleration_{sw}"
    
                features[change_col] = df[col] - df[col].shift(sw)
                features[slope_col] = ( df[col].rolling(sw, min_periods=max(2, int(sw * 0.8))).apply(self._linear_regression_slope, raw=True))
    
                features[acceleration_col] = (features[slope_col] - features[slope_col].shift(sw))
    
        
        return pd.DataFrame(features, index=df.index)


        
    def _build_multi_window_comparison_features(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Adds relationships between short, medium, and long behavior windows.

        Helps the classifier understand local range inside broader range, local compression inside broader balance,
        short-term behavior strength relative to medium-term behavior, one-sided short-term pressure inside a broader range,
        and agreement or disagreement across windows

        This is important because the market can be ranging on 20 candles but trending on 100 candles, compressing locally 
        inside a broader range, or transitioning on the short window while still balanced on the long window.

        This method creates:

            - ratio features
            - difference features
            - position alignment- range component agreement
            - range candidate agreement
            - final range agreement score
         """

        windows = sorted(self.config.windows)

        if len(windows) < 2:

            return pd.DataFrame(index=df.index)

        
        features: dict[str, pd.Series] = {}

        comparison_features = [

            "range_width_atr",
            "directional_efficiency",
            "mid_cross_frequency",
            "touch_balance",
            "rotation_score",
            "flatness_score",
            "two_sided_touch_score",
            "boundary_activity_score",
            "range_behavior_candidate",
            "abs_robust_trendline_move_atr",
            "one_sided_position_pressure",
            "atr_compression_ratio",
        ]

        component_features = [
            "directional_efficiency",
            "flatness_score",
            "rotation_score",
            "two_sided_touch_score",
            "boundary_activity_score"]

        
        for short, long in zip(windows[:-1], windows[1:]):

            for feature in comparison_features:

                short_col = f"{feature}_{short}"
                long_col = f"{feature}_{long}"
                ratio_col = f"{feature}_ratio_{short}_{long}"
                diff_col = f"{feature}_diff_{short}_{long}"

                
                features[ratio_col] = self._make_safe_ratio(
                    df,
                    numerator=short_col,
                    denominator=long_col,
                    clip=10.0,
                )

                
                features[diff_col] = self._make_safe_diff(
                    df,
                    left=short_col,
                    right=long_col,
                )

            short_pos_col = f"position_in_range_{short}"
            long_pos_col = f"position_in_range_{long}"
            pos_diff_col = f"position_in_range_diff_{short}_{long}"
            pos_alignment_col = f"position_alignment_{short}_{long}"

            
            if short_pos_col in df.columns and long_pos_col in df.columns:

                pos_diff = df[short_pos_col] - df[long_pos_col]

                pos_alignment = (1.0 - pos_diff.abs()).clip(0.0, 1.0)

            
            else:

                pos_diff = pd.Series(np.nan, index=df.index)

                pos_alignment = pd.Series(np.nan, index=df.index)

            
            features[pos_diff_col] = pos_diff

            features[pos_alignment_col] = pos_alignment

            agreement_components = []

            
            for component in component_features:

                short_col = f"{component}_{short}"

                long_col = f"{component}_{long}"

                if short_col not in df.columns or long_col not in df.columns:

                    continue

                
                if component == "directional_efficiency":

                    short_component = 1.0 - df[short_col].clip(0.0, 1.0)

                    long_component = 1.0 - df[long_col].clip(0.0, 1.0)

                
                else:

                    short_component = df[short_col].clip(0.0, 1.0)

                    long_component = df[long_col].clip(0.0, 1.0)

                component_agreement = (1.0 - (short_component - long_component).abs()).clip(0.0, 1.0)

                agreement_components.append(component_agreement)

            
            component_agreement_col = f"range_component_agreement_{short}_{long}"

            if agreement_components:

                component_agreement = pd.concat( agreement_components, axis=1).mean(axis=1)

            else:

                component_agreement = pd.Series(np.nan, index=df.index)

            
            features[component_agreement_col] = component_agreement

            candidate_short_col = f"range_behavior_candidate_{short}"
            candidate_long_col = f"range_behavior_candidate_{long}"
            candidate_agreement_col = f"range_candidate_agreement_{short}_{long}"

            
            if candidate_short_col in df.columns and candidate_long_col in df.columns:

                candidate_agreement = ( 1.0 - (df[candidate_short_col] - df[candidate_long_col]).abs()).clip(0.0, 1.0)

            else:

                candidate_agreement = pd.Series(np.nan, index=df.index)

            
            features[candidate_agreement_col] = candidate_agreement

            features[f"range_agreement_{short}_{long}"] = pd.concat(
                [component_agreement, candidate_agreement, pos_alignment], axis=1).mean(axis=1)

        
        return pd.DataFrame(features, index=df.index)



    def _build_volume_context_features(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Adds volume/activity context. Uses bid_volume and ask_volume when available.
        If volume columns are missing it will return empty dataframe.
        """

        if "bid_volume" not in df.columns or "ask_volume" not in df.columns:

            return pd.DataFrame(index=df.index)

    
        c = self.config

        features: dict[str, pd.Series] = {}

        total_volume = df["bid_volume"].astype(float) + df["ask_volume"].astype(float)

        volume_imbalance = ((df["bid_volume"].astype(float) - df["ask_volume"].astype(float)) / (total_volume + c.eps))

        features["total_volume"] = total_volume
        features["volume_imbalance"] = volume_imbalance

    
        for window in sorted(self.config.windows):

            min_periods = self._min_periods(window)

            volume_mean = (total_volume.rolling(window, min_periods=min_periods).mean().shift(1))

            volume_std = (total_volume.rolling(window, min_periods=min_periods).std(ddof=0).shift(1))

            safe_mean = volume_mean.where(volume_mean.abs() > c.eps, np.nan)
            safe_std = volume_std.where(volume_std > c.eps, np.nan)

            volume_ratio = total_volume / safe_mean

            volume_zscore = ((total_volume - volume_mean) / safe_std).clip(lower=-c.zscore_clip, upper=c.zscore_clip)

            features[f"volume_mean_{window}"] = volume_mean
            features[f"volume_ratio_{window}"] = volume_ratio.clip(lower=0.0, upper=10.0)
            features[f"volume_zscore_{window}"] = volume_zscore

            upper_zone_col = f"near_upper_zone_{window}"
            lower_zone_col = f"near_lower_zone_{window}"

            if upper_zone_col in df.columns:

                volume_near_upper = (total_volume.where(df[upper_zone_col] == 1.0, 0.0).rolling(window, min_periods=min_periods).sum()

                    / ( total_volume.rolling(window, min_periods=min_periods).sum() + c.eps))

            
            else:

                volume_near_upper = pd.Series(np.nan, index=df.index)

            
            if lower_zone_col in df.columns:

                volume_near_lower = (total_volume.where(df[lower_zone_col] == 1.0, 0.0).rolling(window, min_periods=min_periods).sum()

                    / (total_volume.rolling(window, min_periods=min_periods).sum()+ c.eps))

            
            else:

                volume_near_lower = pd.Series(np.nan, index=df.index)

            
            features[f"volume_near_upper_share_{window}"] = volume_near_upper
            features[f"volume_near_lower_share_{window}"] = volume_near_lower
            features[f"volume_boundary_imbalance_{window}"] = (volume_near_upper - volume_near_lower)


    
        return pd.DataFrame(features, index=df.index)
        
    # ---------------------------------------------------------------------
    #                            Helper Functions
    # ---------------------------------------------------------------------

    def _min_periods(self, window: int) -> int:

        return max(2, int(window * self.config.min_periods_ratio))


    def _validate_input_schema(self, df: pd.DataFrame) -> None:

        required_columns = ["timestamp", "open", "high", "low", "close"]
    
        missing = [col for col in required_columns if col not in df.columns]
    
        if missing:
    
            raise ValueError(f"RangeFeatureExtractor expected OHLC candle data with columns "
                             f"{required_columns}, but missing: {missing}. "
                             f"Received columns: {list(df.columns)}")

    def _validate_ohlc_data(self, df: pd.DataFrame) -> None:
        
        required = ["open", "high", "low", "close"]
    
        for col in required:
            
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
        if df[required].isna().any().any():
            
            bad_cols = df[required].columns[df[required].isna().any()].tolist()
            
            raise ValueError(f"OHLC columns contain NaN/non-numeric values: {bad_cols}")
    
        bad_rows = df[
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        ]

        
        if len(bad_rows) > 0:
            
            raise ValueError(f"Invalid OHLC data found in {len(bad_rows)} rows. "
                              "Expected high >= open/close/low and low <= open/close/high.")


    def _is_zscore_column(self, col: str) -> bool:

        return any(col.endswith(f"_z{z_window}") for z_window in self.config.zscore_windows)


    def get_model_feature_columns(self, df: pd.DataFrame) -> list[str]:
        
        """
        Returns model-ready feature columns. Excludes timestamps, raw OHLC, raw price boundaries, and internal/debug flags.
        """

        exclude_exact = {
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "true_range",
            "bid_volume",
            "ask_volume",
            "candle_range",
            "body_size",
            "total_volume"
        }

        exclude_keywords = [
            "range_high_",
            "range_low_",
            "range_mid_",
            "quantile_range_high_",
            "quantile_range_low_",
            "quantile_range_mid_",
            "upper_zone_start_",
            "lower_zone_end_",
            "near_upper_zone_",
            "near_lower_zone_",
            "close_slope_",
            "robust_close_slope_",
            "mid_cross_count_",
            "upper_touch_count_",
            "lower_touch_count_",
            "total_touch_count_",
            "weak_" #For when weak labels are implemented
        ]

        selected = []

        for col in df.columns:
            if col in exclude_exact:
                continue

            if any(keyword in col for keyword in exclude_keywords):
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                selected.append(col)

        return selected

    def transform_and_save_training_features(self,
                                             df: pd.DataFrame,
                                             *,
                                             output_dir: str | Path = "feature_data/range_training_features",
                                             symbol: str = "EURUSD",
                                             timeframe: str = "D1",
                                             filename: Optional[str] = None,
                                             overwrite: bool = False) -> pd.DataFrame:
        
        """
        Runs transform(), selects only model-ready training columns,
        saves them as Parquet, and returns the training feature dataframe.
    
        This does not save raw OHLC, timestamps, internal range boundaries,
        debug/helper columns, or weak labels.
        """
    
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
        if training_filename is None:

        training_filename = f"{symbol}_{timeframe}_range_training_features.parquet"

    if full_filename is None:

        full_filename = f"{symbol}_{timeframe}_range_full_features.parquet"

        training_feature_path = output_dir / training_filename
    
        full_feature_path = output_dir / full_filename
    
        existing_files = [ path for path in [training_feature_path, full_feature_path] if path.exists() ]

        
        if existing_files and not overwrite:
    
            existing_paths = "\n".join(str(path) for path in existing_files)
    
            raise FileExistsError(f"Feature file(s) already exist:\n{existing_paths}\n"
                                   "Use overwrite=True if you want to replace them.")

        
        full_features = self.transform(df)
    
        model_cols = self.get_model_feature_columns(full_features)
    
        training_features = full_features[model_cols].copy()
    
        full_features = full_features.replace([np.inf, -np.inf], np.nan)
    
        training_features = training_features.replace([np.inf, -np.inf], np.nan)
    
        full_features.to_parquet(full_feature_path, index=False)
    
        training_features.to_parquet(training_feature_path, index=False)
    
        print(f"[FEATURE SAVE] Saved full features to: {full_feature_path}")
        print(f"[TRAINING FEATURE SAVE] Saved training features to: {training_feature_path}")
        print(f"[FEATURE SAVE] Full feature shape: {full_features.shape}")
        print(f"[TRAINING FEATURE SAVE] Training feature shape: {training_features.shape}")

        
        return training_features