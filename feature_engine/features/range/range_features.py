from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

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
    
        data = self._add_atr(data)
    
        for window in self.config.windows:
            
            data = self._add_range_geometry_features(data, window)
            
            data = self._add_directional_efficiency_features(data, window)
            
            data = self._add_boundary_touch_features(data, window)
            
            data = self._add_rotation_features(data, window)
            
            data = self._add_slope_features(data, window)
            
            data = self._add_lifecycle_features(data, window)

        '''
        ATR context should happen before multi-window comparison because
        multi-window comparison uses atr_compression_ratio_N.
        '''
        
        data = self._add_atr_context_features(data)
    
        #multi window comparison creates range candidates and cross-window agreement
        data = self._add_multi_window_comparison_features(data)

        #all features ready, add rolling z scores
        data = self._add_rolling_zscores(data)

        
        return data

    

    # ---------------------------------------------------------------------
    #                          Rolling Z-SCORE
    # ---------------------------------------------------------------------


    def _zscore_feature_columns(self, df: pd.DataFrame) -> list[str]:

        """
        Selects behavior features that benefit from rolling z-score.
        Avoids raw prices, timestamps, boolean zone flags, and already-zscored columns.
        """
    
        include_keywords = ["range_width_atr_",
                            "directional_efficiency_",
                            "mid_cross_frequency_",
                            "rotation_score_",
                            "touch_balance_",
                            "two_sided_touch_score_",
                            "boundary_activity_score_",
                            "abs_robust_trendline_move_atr_",
                            "flatness_score_",
                            "atr_compression_ratio_",
                            "one_sided_position_pressure_",
                            "range_behavior_candidate_",
                            "range_agreement_",
                            "range_component_agreement_",
                            "range_candidate_agreement_",
                            "slope_outlier_sensitivity_"]
    
        exclude_keywords = ["_z",
                            "range_high_",
                            "range_low_",
                            "range_mid_",
                            "upper_zone_start_",
                            "lower_zone_end_",
                            "near_upper_zone_",
                            "near_lower_zone_",
                            "timestamp"]
    
        
        selected = []
    
        for col in df.columns:
    
            if any(excluded in col for excluded in exclude_keywords):
    
                continue
    
            
            if any(included in col for included in include_keywords):
    
                selected.append(col)
    
        
        return selected


    
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
    #                               Geometry
    # ---------------------------------------------------------------------
    
    def _add_range_geometry_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:

        c = self.config

        min_periods = self._min_periods(window)

        atr_col = f"atr_{c.atr_window}"
        high_col = f"range_high_{window}"
        low_col = f"range_low_{window}"
        mid_col = f"range_mid_{window}"
        width_col = f"range_width_{window}"


        df[high_col] = df["high"].rolling(window, min_periods=min_periods).max()

        df[low_col] = df["low"].rolling(window, min_periods=min_periods).min()

        df[mid_col] = (df[high_col] + df[low_col]) / 2.0

        df[width_col] = df[high_col] - df[low_col]

        df[f"range_width_atr_{window}"] = df[width_col] / (df[atr_col] + c.eps)

        df[f"upper_zone_start_{window}"] = df[high_col] - (df[width_col] * c.zone_pct)

        df[f"lower_zone_end_{window}"] = df[low_col] + (df[width_col] * c.zone_pct)

        df[f"position_in_range_{window}"] = ((df["close"] - df[low_col]) / (df[width_col] + c.eps)).clip(lower=0.0, upper=1.0)

        df[f"distance_to_range_high_{window}"] = df[high_col] - df["close"]

        df[f"distance_to_range_low_{window}"] = df["close"] - df[low_col]

        df[f"distance_to_range_high_atr_{window}"] = (df[f"distance_to_range_high_{window}"] / (df[atr_col] + c.eps))

        df[f"distance_to_range_low_atr_{window}"] = (df[f"distance_to_range_low_{window}"] / (df[atr_col] + c.eps))

        df[f"current_distance_from_mid_{window}"] = ((df["close"] - df[mid_col]).abs() / (df[width_col] + c.eps))


        return df


    # ---------------------------------------------------------------------
    #                              Directional
    # ---------------------------------------------------------------------
    def _add_directional_efficiency_features(self,df: pd.DataFrame, window: int) -> pd.DataFrame:

        c = self.config

        min_periods = self._min_periods(window)

        net_change = (df["close"] - df["close"].shift(window - 1)).abs()

        total_movement = (df["close"].diff().abs().rolling(window, min_periods=min_periods).sum())

        df[f"directional_efficiency_{window}"] = net_change / (total_movement + c.eps)

        return df


    # ---------------------------------------------------------------------
    #                              Rotational
    # ---------------------------------------------------------------------

    def _add_rotation_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:

        c = self.config

        min_periods = self._min_periods(window)

        mid_col = f"range_mid_{window}"
        width_col = f"range_width_{window}"

        #avoid counting the transition from invalid to valid midpoint as a real midpoint cross
        above_mid = (df["close"] > df[mid_col]).where(df[mid_col].notna(), np.nan)

        mid_cross = (above_mid != above_mid.shift(1)).astype(float)

        mid_cross = mid_cross.where(above_mid.notna() & above_mid.shift(1).notna(), np.nan)
        
        df[f"mid_cross_count_{window}"] = (mid_cross.rolling(window, min_periods=min_periods).sum())

        df[f"mid_cross_frequency_{window}"] = df[f"mid_cross_count_{window}"] / window

        df[f"avg_distance_from_mid_{window}"] = (((df["close"] - df[mid_col]).abs() / (df[width_col] + c.eps)).rolling(window,
                                                                                                                       min_periods=min_periods)
                                                                                                                       .mean())
        '''
        Current rotational score:
        ->more midpoint crossing helps
        ->lower average distance from midpoint helps
        '''
        
        df[f"rotation_score_{window}"] = (df[f"mid_cross_frequency_{window}"] * (1.0 - df[f"avg_distance_from_mid_{window}"].clip(0.0, 1.0))
                                         ).clip(lower=0.0, upper=1.0)


        
        return df



    # ---------------------------------------------------------------------
    #                              Lifecycle
    # ---------------------------------------------------------------------
    def _add_lifecycle_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:

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

        sw = self.config.slope_window

        c = self.config

        
        lifecycle_base_cols = [f"range_width_atr_{window}",
                               f"directional_efficiency_{window}",
                               f"touch_balance_{window}",
                               f"mid_cross_frequency_{window}",
                               f"boundary_activity_score_{window}",
                               f"two_sided_touch_score_{window}",
                               f"rotation_score_{window}",
                               f"flatness_score_{window}",
                               f"position_in_range_{window}"]


        
        for col in lifecycle_base_cols:

            if col not in df.columns:

                continue

            
            df[f"{col}_change_{sw}"] = df[col] - df[col].shift(sw)

            df[f"{col}_slope_{sw}"] = (df[col].rolling(sw, min_periods=max(2, int(sw * 0.8))).apply(self._linear_regression_slope, raw=True))

        
        #range-width state candidates which are just numeric descriptors, not final classification labels

        width_col = f"range_width_atr_{window}"

        width_slope_col = f"{width_col}_slope_{sw}"

        df[f"range_expansion_pressure_{window}"] = (df[width_slope_col].clip(lower=0.0))

        df[f"range_compression_pressure_{window}"] = (( -df[width_slope_col]).clip(lower=0.0))

        
        #directional pressure rising while range features weaken can indicate transition
        de_slope_col = f"directional_efficiency_{window}_slope_{sw}"

        df[f"directional_pressure_change_{window}"] = df[de_slope_col]

        #position persistence near upper/lower area.

        pos_col = f"position_in_range_{window}"

        df[f"time_near_upper_{window}"] = ((df[pos_col] >= 1.0 - c.zone_pct).astype(float).rolling(window, min_periods=self._min_periods(window))
                                          .mean())


        
        df[f"time_near_lower_{window}"] = ((df[pos_col] <= c.zone_pct).astype(float).rolling(window, min_periods=self._min_periods(window))
                                          .mean())

        
        df[f"one_sided_position_pressure_{window}"] = (pd.concat([df[f"time_near_upper_{window}"], df[f"time_near_lower_{window}"]],axis=1
                                                                ).max(axis=1))


        
        return df

        
    # ---------------------------------------------------------------------
    #                              Boundaries
    # ---------------------------------------------------------------------
    def _add_boundary_touch_features(self, df: pd.DataFrame, window: int) -> pd.DataFrame:

        c = self.config

        min_periods = self._min_periods(window)

        upper_zone_col = f"upper_zone_start_{window}"
        lower_zone_col = f"lower_zone_end_{window}"
        near_upper_col = f"near_upper_zone_{window}"
        near_lower_col = f"near_lower_zone_{window}"


        df[near_upper_col] = (df["high"] >= df[upper_zone_col]).astype(float)

        df[near_lower_col] = (df["low"] <= df[lower_zone_col]).astype(float)

        upper_touch_col = f"upper_touch_count_{window}"

        lower_touch_col = f"lower_touch_count_{window}"

        df[upper_touch_col] = (df[near_upper_col].rolling(window, min_periods=min_periods).sum())


        df[lower_touch_col] = (df[near_lower_col].rolling(window, min_periods=min_periods).sum())

        df[f"upper_touch_frequency_{window}"] = df[upper_touch_col] / window
        df[f"lower_touch_frequency_{window}"] = df[lower_touch_col] / window
        
        df[f"total_touch_count_{window}"] = (df[upper_touch_col] + df[lower_touch_col])

        df[f"boundary_activity_score_{window}"] = (df[f"total_touch_count_{window}"] / (2.0 * window)).clip(lower=0.0, upper=1.0)

        max_touches = pd.concat([df[upper_touch_col], df[lower_touch_col]],axis=1).max(axis=1)

        min_touches = pd.concat([df[upper_touch_col], df[lower_touch_col]], axis=1).min(axis=1)

        df[f"touch_balance_{window}"] = min_touches / (max_touches + c.eps)

        df[f"two_sided_touch_score_{window}"] = (df[f"touch_balance_{window}"] * df[f"boundary_activity_score_{window}"]).clip(lower=0.0,
                                                                                                                               upper=1.0)


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




    def _add_range_behavior_candidates(self, df: pd.DataFrame) -> pd.DataFrame:

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

        for window in sorted(self.config.windows):

            required_cols = [f"directional_efficiency_{window}",
                             f"flatness_score_{window}",
                             f"rotation_score_{window}",
                             f"two_sided_touch_score_{window}",
                             f"boundary_activity_score_{window}"]

            missing = [col for col in required_cols if col not in df.columns]

            candidate_col = f"range_behavior_candidate_{window}"

            if missing:

                df[candidate_col] = np.nan

                continue

            
            inefficiency = 1.0 - df[f"directional_efficiency_{window}"].clip(0.0, 1.0)

            flatness = df[f"flatness_score_{window}"].clip(0.0, 1.0)

            rotation = df[f"rotation_score_{window}"].clip(0.0, 1.0)

            two_sided = df[f"two_sided_touch_score_{window}"].clip(0.0, 1.0)

            boundary_activity = df[f"boundary_activity_score_{window}"].clip(0.0, 1.0)



            df[candidate_col] = (0.30 * inefficiency + 0.25 * flatness + 0.20 * rotation + 0.15 * two_sided + 0.10 * boundary_activity).clip(0.0,
                                                                                                                                             1.0)

        
        return df




    def _add_multi_window_comparison_features(self, df: pd.DataFrame) -> pd.DataFrame:

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

            return df


        #Compute candidate scores once per window. Avoid repeated calculations inside each pair comparison.

        df = self._add_range_behavior_candidates(df)

        comparison_features = ["range_width_atr",
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
                           "atr_compression_ratio"]

        component_features = ["directional_efficiency",
                          "flatness_score",
                          "rotation_score",
                          "two_sided_touch_score",
                          "boundary_activity_score"]


        for short, long in zip(windows[:-1], windows[1:]):

            # ------------------------------------------------------------
            # Ratios and differences
            # --------------------------------------------------------------

            for feature in comparison_features:

                short_col = f"{feature}_{short}"
                long_col = f"{feature}_{long}"
                ratio_col = f"{feature}_ratio_{short}_{long}"
                diff_col = f"{feature}_diff_{short}_{long}"
        
                self._safe_ratio(df,
                             numerator=short_col,
                             denominator=long_col,
                             output=ratio_col,
                             clip=10.0)

                self._safe_diff(df, left=short_col, right=long_col, output=diff_col)

            # --------------------------------------------------------------
            # Position alignment
            # --------------------------------------------------------------

            short_pos_col = f"position_in_range_{short}"
            long_pos_col = f"position_in_range_{long}"
            pos_diff_col = f"position_in_range_diff_{short}_{long}"
            pos_alignment_col = f"position_alignment_{short}_{long}"


            if short_pos_col in df.columns and long_pos_col in df.columns:

                df[pos_diff_col] = df[short_pos_col] - df[long_pos_col]
                
                '''
                Both position values are clipped 0-1 elsewhere, so abs diff is 0-1.
                Higher alignment means short and long windows place price similarly.
                '''
                
                df[pos_alignment_col] = (1.0 - df[pos_diff_col].abs()).clip(0.0, 1.0)

            else:

                df[pos_diff_col] = np.nan
                df[pos_alignment_col] = np.nan


            # --------------------------------------------------------------
            # Component-level range agreement
            # --------------------------------------------------------------

            agreement_components = []


            for feature in component_features:

                short_col = f"{feature}_{short}"
                long_col = f"{feature}_{long}"

                if short_col not in df.columns or long_col not in df.columns:

                    continue

                
                if feature == "directional_efficiency":

                    #For range agreement, low efficiency is the range-like trait

                    short_component = 1.0 - df[short_col].clip(0.0, 1.0)
                    long_component = 1.0 - df[long_col].clip(0.0, 1.0)

                else:

                    short_component = df[short_col].clip(0.0, 1.0)
                    long_component = df[long_col].clip(0.0, 1.0)

                component_agreement = (1.0 - (short_component - long_component).abs()).clip(0.0, 1.0)

                agreement_components.append(component_agreement)

                component_agreement_col = f"range_component_agreement_{short}_{long}"


            if agreement_components:

                df[component_agreement_col] = pd.concat(agreement_components, axis=1).mean(axis=1)


            else:

                df[component_agreement_col] = np.nan


            # --------------------------------------------------------------
            # Candidate-level agreement
            # -------------------------------------------------------------

            candidate_short_col = f"range_behavior_candidate_{short}"

            candidate_long_col = f"range_behavior_candidate_{long}"

            candidate_agreement_col = f"range_candidate_agreement_{short}_{long}"

            if candidate_short_col in df.columns and candidate_long_col in df.columns:

                df[candidate_agreement_col] = (1.0 - ( df[candidate_short_col] - df[candidate_long_col]).abs()).clip(0.0, 1.0)

            else:

                df[candidate_agreement_col] = np.nan

            # --------------------------------------------------------------
            # Final range agreement
            # --------------------------------------------------------------

            '''
            this is a descriptive feature, not the final classifier output.
            it tells the model whether short and long windows are telling
            a similar range-behavior story.
            '''

            df[f"range_agreement_{short}_{long}"] = pd.concat([df[component_agreement_col],
                                                               df[candidate_agreement_col],
                                                               df[pos_alignment_col]],
                                                              axis=1).mean(axis=1)



        return df


        
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


        