import pandas as pd
import numpy as np
from scipy.stats import rankdata


class Alpha101:
    """
    Implements the 101 Formulaic Alphas described by Zura Kakushadze.
    
    The formulas are directly translated from the paper into Python code.
    Each alpha is a method of this class.
    
    Args:
        df (pd.DataFrame): DataFrame with a MultiIndex (date, asset) and columns
                           for 'open', 'high', 'low', 'close', 'volume', 'vwap', 
                           'returns', and 'sector' (for neutralization).
    """

    def __init__(self, df: pd.DataFrame):
        # Ensure the DataFrame is sorted before any calculations begin.
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
            
        self.df = df
        self.open = df['open']
        self.high = df['high']
        self.low = df['low']
        self.close = df['close']
        self.volume = df['volume']
        self.vwap = df['vwap']
        self.returns = df['returns']
        self.cap = df['cap']
        
        # --- Pre-calculate adv{d} for performance ---
        self.adv5 = self.adv(5)
        self.adv10 = self.adv(10)
        self.adv15 = self.adv(15)
        self.adv20 = self.adv(20)
        self.adv30 = self.adv(30)
        self.adv40 = self.adv(40)
        self.adv50 = self.adv(50)
        self.adv60 = self.adv(60)
        self.adv81 = self.adv(81)
        self.adv120 = self.adv(120)
        self.adv150 = self.adv(150)
        self.adv180 = self.adv(180)
    
    # ----------------------------------------------------------------
    # --- Operator Implementations (from Appendix A) ---
    # ----------------------------------------------------------------

    def delay(self, series, d):
        return series.groupby(level='asset').shift(d)

    def correlation(self, x, y, d):
        return x.groupby(level='asset').rolling(d).corr(y).reset_index(level=0, drop=True)

    def covariance(self, x, y, d):
        return x.groupby(level='asset').rolling(d).cov(y).reset_index(level=0, drop=True)
        
    def rank(self, series):
        # Using pct=True gives percentile ranks, which is more robust.
        return series.groupby(level='date').rank(pct=True)
        
    def delta(self, series, d):
        return series - self.delay(series, d)

    def scale(self, series, a=1):
        sum_abs = series.groupby(level='date').transform(lambda x: x.abs().sum())
        return series * a / sum_abs.replace(0, 1)
    
    def signedpower(self, series, a):
        return np.sign(series) * (np.abs(series) ** a)

    def decay_linear(self, series, d):
        weights = np.arange(1, d + 1)
        # The 'raw=False' is important to handle the Series with its index
        return series.groupby(level='asset').rolling(window=d).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=False).reset_index(level=0, drop=True)
    
    def indneutralize(self, series, group_col='sector'):
        # Demean by industry group
        return series.groupby([series.index.get_level_values('date'), self.df[group_col]]).transform(lambda x: x - x.mean())

    def ts_min(self, series, d):
        return series.groupby(level='asset').rolling(d).min().reset_index(level=0, drop=True)

    def ts_max(self, series, d):
        return series.groupby(level='asset').rolling(d).max().reset_index(level=0, drop=True)
        
    def ts_argmax(self, series, d):
        # rolling().apply() with np.argmax is slow. A more performant version can be written if needed.
        return series.groupby(level='asset').rolling(d).apply(np.argmax, raw=True).reset_index(level=0, drop=True) + 1

    def ts_argmin(self, series, d):
        return series.groupby(level='asset').rolling(d).apply(np.argmin, raw=True).reset_index(level=0, drop=True) + 1

    def ts_rank(self, series, d):
        # Rank of the current value within the past d days
        return series.groupby(level='asset').rolling(d).apply(lambda x: rankdata(x)[-1] / len(x), raw=True).reset_index(level=0, drop=True)

    def sum(self, series, d):
        return series.groupby(level='asset').rolling(d).sum().reset_index(level=0, drop=True)

    def product(self, series, d):
        return series.groupby(level='asset').rolling(d).apply(np.prod, raw=True).reset_index(level=0, drop=True)

    def stddev(self, series, d):
        return series.groupby(level='asset').rolling(d).std().reset_index(level=0, drop=True)
        
    def adv(self, d):
        # The paper defines adv{d} as sum(volume, d)/d, which is a rolling mean
        return self.volume.groupby(level='asset').rolling(d).mean().reset_index(level=0, drop=True)
    
    # Helper for element-wise min/max to avoid confusion with Python's built-in min/max
    def min(self, x, y):
        return np.minimum(x, y)

    def max(self, x, y):
        return np.maximum(x, y)

    # ----------------------------------------------------------------
    # --- Alpha Implementations ---
    # ----------------------------------------------------------------
    
    def alpha001(self):
        """Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)"""
        std_returns = self.stddev(self.returns, 20)
        inner = self.close.mask(self.returns < 0, std_returns)
        return self.rank(self.ts_argmax(self.signedpower(inner, 2.), 5)) - 0.5
    
    def alpha002(self):
        """Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))"""
        x = self.rank(self.delta(np.log(self.volume.replace(0,1)), 2))
        y = self.rank((self.close - self.open) / self.open)
        return -1 * self.correlation(x, y, 6)

    def alpha003(self):
        """Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))"""
        return -1 * self.correlation(self.rank(self.open), self.rank(self.volume), 10)

    def alpha004(self):
        """Alpha#4: (-1 * Ts_Rank(rank(low), 9))"""
        return -1 * self.ts_rank(self.rank(self.low), 9)

    def alpha005(self):
        """Alpha#5: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))"""
        x = self.rank(self.open - (self.sum(self.vwap, 10) / 10))
        y = -1 * abs(self.rank(self.close - self.vwap))
        return x * y

    def alpha006(self):
        """Alpha#6: (-1 * correlation(open, volume, 10))"""
        return -1 * self.correlation(self.open, self.volume, 10)

    def alpha007(self):
        """Alpha#7: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))"""
        delta_close_7 = self.delta(self.close, 7)
        val_if_true = -1 * self.ts_rank(abs(delta_close_7), 60) * np.sign(delta_close_7)
        condition, _ = (self.adv20 < self.volume).align(val_if_true, join='right')
        return val_if_true.where(condition, -1.0)
    
    def alpha008(self):
        """Alpha#8: (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))"""
        adv_term = self.sum(self.open, 5) * self.sum(self.returns, 5)
        return -1 * self.rank(adv_term - self.delay(adv_term, 10))

    def alpha009(self):
        """Alpha#9: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))"""
        delta_close_1 = self.delta(self.close, 1)
        cond1 = self.ts_min(delta_close_1, 5) > 0
        cond2 = self.ts_max(delta_close_1, 5) < 0
        val = (-delta_close_1).copy()
        val[cond1 | cond2] = delta_close_1
        return val

    def alpha010(self):
        """Alpha#10: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))"""
        delta_close_1 = self.delta(self.close, 1)
        cond1 = self.ts_min(delta_close_1, 4) > 0
        cond2 = self.ts_max(delta_close_1, 4) < 0
        val = (-delta_close_1).copy()
        val[cond1 | cond2] = delta_close_1
        return self.rank(val)

    def alpha011(self):
        """Alpha#11: ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))"""
        vwap_minus_close = self.vwap - self.close
        return (self.rank(self.ts_max(vwap_minus_close, 3)) + self.rank(self.ts_min(vwap_minus_close, 3))) * self.rank(self.delta(self.volume, 3))

    def alpha012(self):
        """Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))"""
        return np.sign(self.delta(self.volume, 1)) * (-1 * self.delta(self.close, 1))

    def alpha013(self):
        """Alpha#13: (-1 * rank(covariance(rank(close), rank(volume), 5)))"""
        return -1 * self.rank(self.covariance(self.rank(self.close), self.rank(self.volume), 5))

    def alpha014(self):
        """Alpha#14: ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))"""
        return -1 * self.rank(self.delta(self.returns, 3)) * self.correlation(self.open, self.volume, 10)

    def alpha015(self):
        """Alpha#15: (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))"""
        corr = self.correlation(self.rank(self.high), self.rank(self.volume), 3)
        return -1 * self.sum(self.rank(corr), 3)
    
    def alpha016(self):
        """Alpha#16: (-1 * rank(covariance(rank(high), rank(volume), 5)))"""
        return -1 * self.rank(self.covariance(self.rank(self.high), self.rank(self.volume), 5))

    def alpha017(self):
        """Alpha#17: (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))"""
        vol_ratio = self.volume / self.adv20.replace(0,1)
        return -1 * self.rank(self.ts_rank(self.close, 10)) * self.rank(self.delta(self.delta(self.close, 1), 1)) * self.rank(self.ts_rank(vol_ratio, 5))
    
    def alpha018(self):
        """Alpha#18: (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))"""
        close_minus_open = self.close - self.open
        term1 = self.stddev(abs(close_minus_open), 5)
        term2 = self.correlation(self.close, self.open, 10)
        return -1 * self.rank(term1 + close_minus_open + term2)

    def alpha019(self):
        """Alpha#19: ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))"""
        # Note: (close - delay(close, 7)) is the same as delta(close, 7)
        delta_close_7 = self.delta(self.close, 7)
        return -1 * np.sign(delta_close_7 + delta_close_7) * (1 + self.rank(1 + self.sum(self.returns, 250)))
    
    def alpha020(self):
        """Alpha#20: (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))"""
        r1 = self.rank(self.open - self.delay(self.high, 1))
        r2 = self.rank(self.open - self.delay(self.close, 1))
        r3 = self.rank(self.open - self.delay(self.low, 1))
        return -1 * r1 * r2 * r3

    def alpha021(self):
        """Alpha#21: ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? -1 : ((((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) | ((volume / adv20) == 1)) ? 1 : -1))))"""
        sma8 = self.sum(self.close, 8) / 8
        std8 = self.stddev(self.close, 8)
        sma2 = self.sum(self.close, 2) / 2
        vol_ratio = self.volume / self.adv20.replace(0,1)
        cond1 = (sma8 + std8) < sma2
        cond2 = sma2 < (sma8 - std8)
        cond3 = vol_ratio >= 1
        val = pd.Series(-1.0, index=self.df.index)
        val = val.where(~cond1, -1.0)
        val = val.mask(~cond1, 1.0)
        inner_val = pd.Series(1.0, index=self.df.index).where(cond3, -1.0)
        val = val.mask(~(cond1 | cond2), inner_val)
        return val

    def alpha022(self):
        """Alpha#22: (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))"""
        corr = self.correlation(self.high, self.volume, 5)
        return -1 * self.delta(corr, 5) * self.rank(self.stddev(self.close, 20))

    def alpha023(self):
        """Alpha#23: (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)"""
        cond = (self.sum(self.high, 20) / 20) < self.high
        val = -1 * self.delta(self.high, 2)
        cond, _ = cond.align(val, join='right')
        return pd.Series(0.0, index=val.index).where(~cond, val)

    def alpha024(self):
        """Alpha#24: ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) | ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3)))"""
        sma100 = self.sum(self.close, 100) / 100
        ratio = self.delta(sma100, 100) / self.delay(self.close, 100)
        cond = ratio <= 0.05
        val_if_true = -1 * (self.close - self.ts_min(self.close, 100))
        val_if_false = -1 * self.delta(self.close, 3)
        return val_if_false.where(~cond, val_if_true)

    def alpha025(self):
        """Alpha#25: rank(((((-1 * returns) * adv20) * vwap) * (high - close)))"""
        return self.rank(-1 * self.returns * self.adv20 * self.vwap * (self.high - self.close))

    def alpha026(self):
        """Alpha#26: (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))"""
        corr = self.correlation(self.ts_rank(self.volume, 5), self.ts_rank(self.high, 5), 5)
        return -1 * self.ts_max(corr, 3)

    def alpha027(self):
        """Alpha#27: ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? -1 : 1)"""
        corr = self.correlation(self.rank(self.volume), self.rank(self.vwap), 6)
        cond = 0.5 < self.rank(self.sum(corr, 2) / 2.0)
        val = pd.Series(1.0, index=cond.index)
        return val.where(~cond, -1.0)

    def alpha028(self):
        """Alpha#28: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))"""
        corr = self.correlation(self.adv20, self.low, 5)
        return self.scale(corr + (self.high + self.low) / 2 - self.close)

    def alpha029(self):
        """Alpha#29: (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))"""
        p1 = self.rank(self.delta(self.close - 1, 5))
        p2 = self.rank(self.rank(-1 * p1))
        p3 = self.ts_min(p2, 2)
        p4 = self.scale(np.log(self.sum(p3, 1).clip(lower=1e-6)))
        p5 = self.rank(self.rank(p4))
        term1 = self.product(p5, 1)
        term2 = self.ts_rank(self.delay(-1 * self.returns, 6), 5)
        return self.min(term1, 5) + term2
        
    def alpha030(self):
        """Alpha#30: (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))"""
        d1 = np.sign(self.delta(self.close, 1))
        d2 = np.sign(self.delay(self.delta(self.close, 1), 1))
        d3 = np.sign(self.delay(self.delta(self.close, 1), 2))
        rank_of_signs = self.rank(d1 + d2 + d3)
        return ((1.0 - rank_of_signs) * self.sum(self.volume, 5)) / self.sum(self.volume, 20).replace(0,1)
    
    def alpha031(self):
        """Alpha#31: ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 * delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))"""
        p1 = self.rank(self.rank(self.delta(self.close, 10)))
        p2 = self.decay_linear(-1 * self.rank(p1), 10)
        p3 = self.rank(self.rank(self.rank(p2)))
        p4 = self.rank(-1 * self.delta(self.close, 3))
        p5 = np.sign(self.scale(self.correlation(self.adv20, self.low, 12)))
        return p3 + p4 + p5

    def alpha032(self):
        """Alpha#32: (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))"""
        p1 = self.scale((self.sum(self.close, 7) / 7) - self.close)
        p2 = 20 * self.scale(self.correlation(self.vwap, self.delay(self.close, 5), 230))
        return p1 + p2

    def alpha033(self):
        """Alpha#33: rank((-1 * ((1 - (open / close))^1)))"""
        return self.rank(-1 * (1 - (self.open / self.close.replace(0,1))))

    def alpha034(self):
        """Alpha#34: rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))"""
        p1 = self.rank(self.stddev(self.returns, 2) / self.stddev(self.returns, 5).replace(0,1))
        p2 = self.rank(self.delta(self.close, 1))
        return self.rank((1 - p1) + (1 - p2))

    def alpha035(self):
        """Alpha#35: ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))"""
        p1 = self.ts_rank(self.volume, 32)
        p2 = 1 - self.ts_rank((self.close + self.high) - self.low, 16)
        p3 = 1 - self.ts_rank(self.returns, 32)
        return p1 * p2 * p3

    def alpha036(self):
        """Alpha#36: (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open - close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap, adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))"""
        p1 = self.rank(self.correlation(self.close - self.open, self.delay(self.volume, 1), 15))
        p2 = self.rank(self.open - self.close)
        p3 = self.rank(self.ts_rank(self.delay(-1 * self.returns, 6), 5))
        p4 = self.rank(abs(self.correlation(self.vwap, self.adv20, 6)))
        p5 = self.rank((((self.sum(self.close, 200) / 200) - self.open) * (self.close - self.open)))
        return (2.21 * p1) + (0.7 * p2) + (0.73 * p3) + p4 + (0.6 * p5)

    def alpha037(self):
        """Alpha#37: (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))"""
        return self.rank(self.correlation(self.delay(self.open - self.close, 1), self.close, 200)) + self.rank(self.open - self.close)
    
    def alpha038(self):
        """Alpha#38: ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))"""
        return -1 * self.rank(self.ts_rank(self.close, 10)) * self.rank(self.close / self.open.replace(0,1))

    def alpha039(self):
        """Alpha#39: ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))"""
        p1 = self.rank(self.decay_linear(self.volume / self.adv20.replace(0,1), 9))
        p2 = self.rank(self.delta(self.close, 7) * (1 - p1))
        p3 = 1 + self.rank(self.sum(self.returns, 250))
        return -1 * p2 * p3

    def alpha040(self):
        """Alpha#40: ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))"""
        return -1 * self.rank(self.stddev(self.high, 10)) * self.correlation(self.high, self.volume, 10)
    
    def alpha041(self):
        """Alpha#41: (((high * low)^0.5) - vwap)"""
        return (self.high * self.low)**0.5 - self.vwap

    def alpha042(self):
        """Alpha#42: (rank((vwap - close)) / rank((vwap + close)))"""
        return self.rank(self.vwap - self.close) / self.rank(self.vwap + self.close)

    def alpha043(self):
        """Alpha#43: (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))"""
        return self.ts_rank(self.volume / self.adv20.replace(0,1), 20) * self.ts_rank(-1 * self.delta(self.close, 7), 8)

    def alpha044(self):
        """Alpha#44: (-1 * correlation(high, rank(volume), 5))"""
        return -1 * self.correlation(self.high, self.rank(self.volume), 5)

    def alpha045(self):
        """Alpha#45: (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))"""
        p1 = self.rank(self.sum(self.delay(self.close, 5), 20) / 20)
        p2 = self.correlation(self.close, self.volume, 2)
        p3 = self.rank(self.correlation(self.sum(self.close, 5), self.sum(self.close, 20), 2))
        return -1 * p1 * p2 * p3
        
    def alpha046(self):
        """Alpha#46: ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ? -1 : ( ((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 : ((-1 * 1) * (close - delay(close, 1)))))"""
        p1 = ((self.delay(self.close, 20) - self.delay(self.close, 10)) / 10)
        p2 = ((self.delay(self.close, 10) - self.close) / 10)
        diff = p1 - p2
        val = -1 * (self.close - self.delay(self.close, 1))
        val[diff > 0.25] = -1.0
        val[(diff > 0.25) == False] = 1.0
        return val

    def alpha047(self):
        """Alpha#47: ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))"""
        p1 = (self.rank(1 / self.close) * self.volume) / self.adv20.replace(0,1)
        p2 = (self.high * self.rank(self.high - self.close)) / (self.sum(self.high, 5) / 5).replace(0,1)
        p3 = self.rank(self.vwap - self.delay(self.vwap, 5))
        return p1 * p2 - p3
        
    def alpha048(self):
        """Alpha#48: (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))"""
        corr = self.correlation(self.delta(self.close, 1), self.delta(self.delay(self.close, 1), 1), 250)
        num = self.indneutralize((corr * self.delta(self.close, 1)) / self.close.replace(0,1), 'sector')
        den = self.sum(((self.delta(self.close, 1) / self.delay(self.close, 1).replace(0,1))**2), 250)
        return num / den.replace(0,1)

    def alpha049(self):
        """Alpha#49: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))"""
        p1 = (self.delay(self.close, 20) - self.delay(self.close, 10)) / 10
        p2 = (self.delay(self.close, 10) - self.close) / 10
        diff = p1 - p2
        cond = diff < -0.1
        return (-1 * (self.close - self.delay(self.close, 1))).where(~cond, 1.0)

    def alpha050(self):
        """Alpha#50: (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))"""
        corr = self.correlation(self.rank(self.volume), self.rank(self.vwap), 5)
        return -1 * self.ts_max(self.rank(corr), 5)
        
    def alpha051(self):
        """Alpha#51: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))"""
        p1 = (self.delay(self.close, 20) - self.delay(self.close, 10)) / 10
        p2 = (self.delay(self.close, 10) - self.close) / 10
        diff = p1 - p2
        cond = diff < -0.05
        return (-1 * (self.close - self.delay(self.close, 1))).where(~cond, 1.0)

    def alpha052(self):
        """Alpha#52: ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))"""
        p1 = -self.ts_min(self.low, 5) + self.delay(self.ts_min(self.low, 5), 5)
        p2 = self.rank((self.sum(self.returns, 240) - self.sum(self.returns, 20)) / 220)
        p3 = self.ts_rank(self.volume, 5)
        return p1 * p2 * p3

    def alpha053(self):
        """Alpha#53: (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))"""
        inner = ((self.close - self.low) - (self.high - self.close)) / (self.close - self.low).replace(0, 0.0001)
        return -1 * self.delta(inner, 9)

    def alpha054(self):
        """Alpha#54: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))"""
        num = -1 * (self.low - self.close) * (self.open**5)
        den = (self.low - self.high) * (self.close**5)
        return num / den.replace(0, 0.0001)

    def alpha055(self):
        """Alpha#55: (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))"""
        num = self.close - self.ts_min(self.low, 12)
        den = self.ts_max(self.high, 12) - self.ts_min(self.low, 12)
        p1 = self.rank(num / den.replace(0, 0.0001))
        return -1 * self.correlation(p1, self.rank(self.volume), 6)

    def alpha056(self):
        """Alpha#56: (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))"""
        sum_ret_2 = self.sum(self.returns, 2)
        p1 = self.rank(self.sum(self.returns, 10) / self.sum(sum_ret_2, 3).replace(0,1))
        p2 = self.rank(self.returns * self.cap)
        return -1 * p1 * p2

    def alpha057(self):
        """Alpha#57: (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))"""
        decay = self.decay_linear(self.rank(self.ts_argmax(self.close, 30)), 2)
        return -1 * (self.close - self.vwap) / decay.replace(0, 0.0001)

    def alpha058(self):
        """Alpha#58: (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume, 3.92795), 7.89291), 5.50322))"""
        neut_vwap = self.indneutralize(self.vwap, 'sector')
        corr = self.correlation(neut_vwap, self.volume, int(3.92795))
        decay = self.decay_linear(corr, int(7.89291))
        return -1 * self.ts_rank(decay, int(5.50322))

    def alpha059(self):
        """Alpha#59: (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap * (1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))"""
        # The weighted vwap is just vwap. This might be a placeholder for a more complex model.
        neut_vwap = self.indneutralize(self.vwap, 'sector') # Using sector for industry
        corr = self.correlation(neut_vwap, self.volume, int(4.25197))
        decay = self.decay_linear(corr, int(16.2289))
        return -1 * self.ts_rank(decay, int(8.19648))

    def alpha060(self):
        """Alpha#60: (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10))))))"""
        p1_num = (self.close - self.low) - (self.high - self.close)
        p1_den = (self.high - self.low).replace(0, 0.0001)
        p1 = self.rank((p1_num / p1_den) * self.volume)
        p2 = self.rank(self.ts_argmax(self.close, 10))
        return -1 * (2 * self.scale(p1) - self.scale(p2))

    def alpha061(self):
        """Alpha#61: (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))"""
        p1 = self.rank(self.vwap - self.ts_min(self.vwap, int(16.1219)))
        p2 = self.rank(self.correlation(self.vwap, self.adv180, int(17.9282)))
        p1, p2 = p1.align(p2, join='inner') # Align before comparing
        return (p1 < p2).astype(float)

    def alpha062(self):
        """Alpha#62: ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)"""
        # (rank(open) + rank(open)) is just 2 * rank(open)
        p1 = self.rank(self.correlation(self.vwap, self.sum(self.adv20, int(22.4101)), int(9.91009)))
        p2_left = 2 * self.rank(self.open)
        p2_right = self.rank((self.high + self.low) / 2) + self.rank(self.high)
        p2 = self.rank((p2_left < p2_right).astype(float))
        return (p1 < p2).astype(float) * -1

    def alpha063(self):
        """Alpha#63: ((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237)) - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180, 37.2467), 13.557), 12.2883))) * -1)"""
        p1_delta = self.delta(self.indneutralize(self.close, 'sector'), int(2.25164))
        p1 = self.rank(self.decay_linear(p1_delta, int(8.22237)))
        
        p2_inner_1 = (self.vwap * 0.318108) + (self.open * (1 - 0.318108))
        p2_inner_2 = self.sum(self.adv180, int(37.2467))
        p2_corr = self.correlation(p2_inner_1, p2_inner_2, int(13.557))
        p2 = self.rank(self.decay_linear(p2_corr, int(12.2883)))
        return -1 * (p1 - p2)

    def alpha064(self):
        """Alpha#64: ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054), sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))), 3.69741))) * -1)"""
        p1_inner_1 = self.sum((self.open * 0.178404) + (self.low * (1 - 0.178404)), int(12.7054))
        p1_inner_2 = self.sum(self.adv120, int(12.7054))
        p1 = self.rank(self.correlation(p1_inner_1, p1_inner_2, int(16.6208)))
        
        p2_inner = (((self.high + self.low) / 2) * 0.178404) + (self.vwap * (1 - 0.178404))
        p2 = self.rank(self.delta(p2_inner, int(3.69741)))
        
        return -1 * (p1 < p2).astype(float)
        
    def alpha065(self):
        """Alpha#65: ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60, 8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)"""
        p1_inner_1 = (self.open * 0.00817205) + (self.vwap * (1 - 0.00817205))
        p1_inner_2 = self.sum(self.adv60, int(8.6911))
        p1 = self.rank(self.correlation(p1_inner_1, p1_inner_2, int(6.40374)))
        p2 = self.rank(self.open - self.ts_min(self.open, int(13.635)))
        return -1 * (p1 < p2).astype(float)

    def alpha066(self):
        """Alpha#66: ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)"""
        p1 = self.rank(self.decay_linear(self.delta(self.vwap, int(3.51013)), int(7.23052)))
        p2_num = self.low - self.vwap
        p2_den = (self.open - (self.high + self.low) / 2).replace(0, 0.0001)
        p2_decay = self.decay_linear(p2_num / p2_den, int(11.4157))
        p2 = self.ts_rank(p2_decay, int(6.72611))
        return -1 * (p1 + p2)

    def alpha067(self):
        """Alpha#67: ((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap, IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)"""
        p1 = self.rank(self.high - self.ts_min(self.high, int(2.14593)))
        p2_neut_vwap = self.indneutralize(self.vwap, 'sector')
        p2_neut_adv20 = self.indneutralize(self.adv20, 'sector') # Using same level for simplicity
        p2 = self.rank(self.correlation(p2_neut_vwap, p2_neut_adv20, int(6.02936)))
        return -1 * (p1 ** p2)

    def alpha068(self):
        """Alpha#68: ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)"""
        p1_corr = self.correlation(self.rank(self.high), self.rank(self.adv15), int(8.91644))
        p1 = self.ts_rank(p1_corr, int(13.9333))
        p2_inner = (self.close * 0.518371) + (self.low * (1 - 0.518371))
        p2 = self.rank(self.delta(p2_inner, int(1.06157)))
        return -1 * (p1 < p2).astype(float)

    def alpha069(self):
        """Alpha#69: ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412), 4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416), 9.0615)) * -1)"""
        p1_delta = self.delta(self.indneutralize(self.vwap, 'sector'), int(2.72412))
        p1 = self.rank(self.ts_max(p1_delta, int(4.79344)))
        p2_inner = (self.close * 0.490655) + (self.vwap * (1 - 0.490655))
        p2_corr = self.correlation(p2_inner, self.adv20, int(4.92416))
        p2 = self.ts_rank(p2_corr, int(9.0615))
        return -1 * (p1 ** p2)

    def alpha070(self):
        """Alpha#70: ((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close, IndClass.industry), adv50, 17.8256), 17.9171)) * -1)"""
        p1 = self.rank(self.delta(self.vwap, int(1.29456)))
        p2_corr = self.correlation(self.indneutralize(self.close, 'sector'), self.adv50, int(17.8256))
        p2 = self.ts_rank(p2_corr, int(17.9171))
        return -1 * (p1 ** p2)

    def alpha071(self):
        """Alpha#71: max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388))"""
        p1_corr = self.correlation(self.ts_rank(self.close, int(3.43976)), self.ts_rank(self.adv180, int(12.0647)), int(18.0175))
        p1_decay = self.decay_linear(p1_corr, int(4.20501))
        p1 = self.ts_rank(p1_decay, int(15.6948))
        
        p2_inner = self.rank((self.low + self.open) - (2 * self.vwap)) ** 2
        p2_decay = self.decay_linear(p2_inner, int(16.4662))
        p2 = self.ts_rank(p2_decay, int(4.4388))
        return self.max(p1, p2)

    def alpha072(self):
        """Alpha#72: (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011)))"""
        p1_corr = self.correlation((self.high + self.low) / 2, self.adv40, int(8.93345))
        p1 = self.rank(self.decay_linear(p1_corr, int(10.1519)))
        p2_corr = self.correlation(self.ts_rank(self.vwap, int(3.72469)), self.ts_rank(self.volume, int(18.5188)), int(6.86671))
        p2 = self.rank(self.decay_linear(p2_corr, int(2.95011)))
        return p1 / p2.replace(0,1)

    def alpha073(self):
        """Alpha#73: (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)), Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)"""
        p1 = self.rank(self.decay_linear(self.delta(self.vwap, int(4.72775)), int(2.91864)))
        
        p2_inner_base = (self.open * 0.147155) + (self.low * (1 - 0.147155))
        p2_inner = self.delta(p2_inner_base, int(2.03608)) / p2_inner_base.replace(0, 0.0001)
        p2_decay = self.decay_linear(p2_inner * -1, int(3.33829))
        p2 = self.ts_rank(p2_decay, int(16.7411))
        
        return -1 * self.max(p1, p2)

    def alpha074(self):
        """Alpha#74: ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))) * -1)"""
        p1 = self.rank(self.correlation(self.close, self.sum(self.adv30, int(37.4843)), int(15.1365)))
        p2_inner = self.rank((self.high * 0.0261661) + (self.vwap * (1 - 0.0261661)))
        p2 = self.rank(self.correlation(p2_inner, self.rank(self.volume), int(11.4791)))
        return -1 * (p1 < p2).astype(float)
        
    def alpha075(self):
        """Alpha#75: (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50), 12.4413)))"""
        p1 = self.rank(self.correlation(self.vwap, self.volume, int(4.24304)))
        p2 = self.rank(self.correlation(self.rank(self.low), self.rank(self.adv50), int(12.4413)))
        return (p1 < p2).astype(float)
    
    def alpha076(self):
        """Alpha#76: (max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)), Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81, 8.14941), 19.569), 17.1543), 19.383)) * -1)"""
        p1 = self.rank(self.decay_linear(self.delta(self.vwap, int(1.24383)), int(11.8259)))
        
        p2_corr = self.correlation(self.indneutralize(self.low, 'sector'), self.adv81, int(8.14941))
        p2_ts_rank = self.ts_rank(p2_corr, int(19.569))
        p2_decay = self.decay_linear(p2_ts_rank, int(17.1543))
        p2 = self.ts_rank(p2_decay, int(19.383))
        
        return -1 * self.max(p1, p2)

    def alpha077(self):
        """Alpha#77: min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)), rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))"""
        p1_inner = (((self.high + self.low) / 2) + self.high) - (self.vwap + self.high)
        p1 = self.rank(self.decay_linear(p1_inner, int(20.0451)))
        
        p2_corr = self.correlation((self.high + self.low) / 2, self.adv40, int(3.1614))
        p2 = self.rank(self.decay_linear(p2_corr, int(5.64125)))
        
        return self.min(p1, p2)

    def alpha078(self):
        """Alpha#78: (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428), sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))"""
        p1_inner1 = self.sum((self.low * 0.352233) + (self.vwap * (1 - 0.352233)), int(19.7428))
        p1_inner2 = self.sum(self.adv40, int(19.7428))
        p1 = self.rank(self.correlation(p1_inner1, p1_inner2, int(6.83313)))
        
        p2 = self.rank(self.correlation(self.rank(self.vwap), self.rank(self.volume), int(5.77492)))
        return p1 ** p2

    def alpha079(self):
        """Alpha#79: (rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))), IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150, 9.18637), 14.6644)))"""
        p1_inner = self.indneutralize((self.close * 0.60733) + (self.open * (1-0.60733)), 'sector')
        p1 = self.rank(self.delta(p1_inner, int(1.23438)))
        
        p2_corr = self.correlation(self.ts_rank(self.vwap, int(3.60973)), self.ts_rank(self.adv150, int(9.18637)), int(14.6644))
        p2 = self.rank(p2_corr)
        return (p1 < p2).astype(float)
        
    def alpha080(self):
        """Alpha#80: ((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))), IndClass.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)"""
        p1_inner = self.indneutralize((self.open * 0.868128) + (self.high * (1 - 0.868128)), 'sector')
        p1 = self.rank(np.sign(self.delta(p1_inner, int(4.04545))))
        
        p2_corr = self.correlation(self.high, self.adv10, int(5.11456))
        p2 = self.ts_rank(p2_corr, int(5.53756))
        
        return -1 * (p1 ** p2)
        
    def alpha081(self):
        """Alpha#81: ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054), 8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)"""
        p1_corr = self.correlation(self.vwap, self.sum(self.adv10, int(49.6054)), int(8.47743))
        p1_inner = self.rank(self.rank(p1_corr)**4)
        p1_prod = self.product(p1_inner, int(14.9655))
        p1 = self.rank(np.log(p1_prod.clip(lower=1e-6)))

        p2 = self.rank(self.correlation(self.rank(self.vwap), self.rank(self.volume), int(5.07914)))
        return -1 * (p1 < p2).astype(float)
        
    def alpha082(self):
        """Alpha#82: (min(rank(decay_linear(delta(open, 1.46063), 14.8717)), Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * 0.634196) + (open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)"""
        p1 = self.rank(self.decay_linear(self.delta(self.open, int(1.46063)), int(14.8717)))
        
        p2_corr = self.correlation(self.indneutralize(self.volume, 'sector'), self.open, int(17.4842))
        p2_decay = self.decay_linear(p2_corr, int(6.92131))
        p2 = self.ts_rank(p2_decay, int(13.4283))
        
        return -1 * self.min(p1, p2)

    def alpha083(self):
        """Alpha#83: ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))"""
        p1_inner = (self.high - self.low) / (self.sum(self.close, 5) / 5).replace(0, 0.0001)
        p1 = self.rank(self.delay(p1_inner, 2)) * self.rank(self.rank(self.volume))
        p2 = (p1_inner / (self.vwap - self.close).replace(0, 0.0001))
        return p1 / p2.replace(0,1)

    def alpha084(self):
        """Alpha#84: SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))"""
        p1 = self.ts_rank(self.vwap - self.ts_max(self.vwap, int(15.3217)), int(20.7127))
        p2 = self.delta(self.close, int(4.96796))
        return self.signedpower(p1, p2)
        
    def alpha085(self):
        """Alpha#85: (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595), 7.11408)))"""
        p1_inner = (self.high * 0.876703) + (self.close * (1 - 0.876703))
        p1 = self.rank(self.correlation(p1_inner, self.adv30, int(9.61331)))
        
        p2_corr = self.correlation(self.ts_rank((self.high + self.low)/2, int(3.70596)), self.ts_rank(self.volume, int(10.1595)), int(7.11408))
        p2 = self.rank(p2_corr)
        return p1 ** p2

    def alpha086(self):
        """Alpha#86: ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open + close) - (vwap + open)))) * -1)"""
        p1_corr = self.correlation(self.close, self.sum(self.adv20, int(14.7444)), int(6.00049))
        p1 = self.ts_rank(p1_corr, int(20.4195))
        p2 = self.rank((self.open + self.close) - (self.vwap + self.open))
        return -1 * (p1 < p2).astype(float)

    def alpha087(self):
        """Alpha#87: (max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))), 1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81, IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)"""
        p1_inner = (self.close * 0.369701) + (self.vwap * (1 - 0.369701))
        p1_delta = self.delta(p1_inner, int(1.91233))
        p1 = self.rank(self.decay_linear(p1_delta, int(2.65461)))
        
        p2_corr = self.correlation(self.indneutralize(self.adv81, 'sector'), self.close, int(13.4132))
        p2_decay = self.decay_linear(abs(p2_corr), int(4.89768))
        p2 = self.ts_rank(p2_decay, int(14.4535))
        return -1 * self.max(p1, p2)
        
    def alpha088(self):
        """Alpha#88: min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957))"""
        p1_inner = (self.rank(self.open) + self.rank(self.low)) - (self.rank(self.high) + self.rank(self.close))
        p1 = self.rank(self.decay_linear(p1_inner, int(8.06882)))

        p2_corr = self.correlation(self.ts_rank(self.close, int(8.44728)), self.ts_rank(self.adv60, int(20.6966)), int(8.01266))
        p2_decay = self.decay_linear(p2_corr, int(6.65053))
        p2 = self.ts_rank(p2_decay, int(2.61957))
        return self.min(p1, p2)

    def alpha089(self):
        """Alpha#89: (Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10, 6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap, IndClass.industry), 3.48158), 10.1466), 15.3012))"""
        p1_corr = self.correlation(self.low, self.adv10, int(6.94279))
        p1_decay = self.decay_linear(p1_corr, int(5.51607))
        p1 = self.ts_rank(p1_decay, int(3.79744))
        
        p2_delta = self.delta(self.indneutralize(self.vwap, 'sector'), int(3.48158))
        p2_decay = self.decay_linear(p2_delta, int(10.1466))
        p2 = self.ts_rank(p2_decay, int(15.3012))
        return p1 - p2

    def alpha090(self):
        """Alpha#90: ((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40, IndClass.subindustry), low, 5.38375), 3.21856)) * -1)"""
        p1 = self.rank(self.close - self.ts_max(self.close, int(4.66719)))
        p2_corr = self.correlation(self.indneutralize(self.adv40, 'sector'), self.low, int(5.38375))
        p2 = self.ts_rank(p2_corr, int(3.21856))
        return -1 * (p1 ** p2)

    def alpha091(self):
        """Alpha#91: ((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close, IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) - rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)"""
        p1_corr = self.correlation(self.indneutralize(self.close, 'sector'), self.volume, int(9.74928))
        p1_decay1 = self.decay_linear(p1_corr, int(16.398))
        p1_decay2 = self.decay_linear(p1_decay1, int(3.83219))
        p1 = self.ts_rank(p1_decay2, int(4.8667))
        
        p2_corr = self.correlation(self.vwap, self.adv30, int(4.01303))
        p2 = self.rank(self.decay_linear(p2_corr, int(2.6809)))
        return -1 * (p1 - p2)

    def alpha092(self):
        """Alpha#92: min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221), 18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024), 6.80584))"""
        p1_inner = ((((self.high + self.low) / 2) + self.close) < (self.low + self.open)).astype(float)
        p1_decay = self.decay_linear(p1_inner, int(14.7221))
        p1 = self.ts_rank(p1_decay, int(18.8683))
        
        p2_corr = self.correlation(self.rank(self.low), self.rank(self.adv30), int(7.58555))
        p2_decay = self.decay_linear(p2_corr, int(6.94024))
        p2 = self.ts_rank(p2_decay, int(6.80584))
        return self.min(p1, p2)

    def alpha093(self):
        """Alpha#93: (Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81, 17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 - 0.524434))), 2.77377), 16.2664)))"""
        p1_corr = self.correlation(self.indneutralize(self.vwap, 'sector'), self.adv81, int(17.4193))
        p1_decay = self.decay_linear(p1_corr, int(19.848))
        p1 = self.ts_rank(p1_decay, int(7.54455))
        p2_inner = (self.close * 0.524434) + (self.vwap * (1 - 0.524434))
        p2_delta = self.delta(p2_inner, int(2.77377))
        p2 = self.rank(self.decay_linear(p2_delta, int(16.2664)))
        return p1 / p2.replace(0,1)

    def alpha094(self):
        """Alpha#94: ((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap, 19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)"""
        p1 = self.rank(self.vwap - self.ts_min(self.vwap, int(11.5783)))
        p2_corr = self.correlation(self.ts_rank(self.vwap, int(19.6462)), self.ts_rank(self.adv60, int(4.02992)), int(18.0926))
        p2 = self.ts_rank(p2_corr, int(2.70756))
        return -1 * (p1 ** p2)
        
    def alpha095(self):
        """Alpha#95: (rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low) / 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))"""
        p1 = self.rank(self.open - self.ts_min(self.open, int(12.4105)))
        p2_inner1 = self.sum((self.high + self.low) / 2, int(19.1351))
        p2_inner2 = self.sum(self.adv40, int(19.1351))
        p2_corr_rank = self.rank(self.correlation(p2_inner1, p2_inner2, int(12.8742)))
        p2 = self.ts_rank(p2_corr_rank**5, int(11.7584))
        p1, p2 = p1.align(p2, join='inner')
        return (p1 < p2).astype(float)
        
    def alpha096(self):
        """Alpha#96: (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404), Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)"""
        p1_corr = self.correlation(self.rank(self.vwap), self.rank(self.volume), int(3.83878))
        p1_decay = self.decay_linear(p1_corr, int(4.16783))
        p1 = self.ts_rank(p1_decay, int(8.38151))
        
        p2_ts_corr = self.correlation(self.ts_rank(self.close, int(7.45404)), self.ts_rank(self.adv60, int(4.13242)), int(3.65459))
        p2_ts_argmax = self.ts_argmax(p2_ts_corr, int(12.6556))
        p2_decay = self.decay_linear(p2_ts_argmax, int(14.0365))
        p2 = self.ts_rank(p2_decay, int(13.4143))
        return -1 * self.max(p1, p2)

    def alpha097(self):
        """Alpha#97: ((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))), IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low, 7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)"""
        p1_inner = self.indneutralize((self.low * 0.721001) + (self.vwap * (1 - 0.721001)), 'sector')
        p1_delta = self.delta(p1_inner, int(3.3705))
        p1 = self.rank(self.decay_linear(p1_delta, int(20.4523)))
        
        p2_ts_corr = self.correlation(self.ts_rank(self.low, int(7.87871)), self.ts_rank(self.adv60, int(17.255)), int(4.97547))
        p2_ts_rank = self.ts_rank(p2_ts_corr, int(18.5925))
        p2_decay = self.decay_linear(p2_ts_rank, int(15.7152))
        p2 = self.ts_rank(p2_decay, int(6.71659))
        return -1 * (p1 - p2)

    def alpha098(self):
        """Alpha#98: (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) - rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206)))"""
        p1_corr = self.correlation(self.vwap, self.sum(self.adv5, int(26.4719)), int(4.58418))
        p1 = self.rank(self.decay_linear(p1_corr, int(7.18088)))
        
        p2_inner_corr = self.correlation(self.rank(self.open), self.rank(self.adv15), int(20.8187))
        p2_ts_argmin = self.ts_argmin(p2_inner_corr, int(8.62571))
        p2_ts_rank = self.ts_rank(p2_ts_argmin, int(6.95668))
        p2 = self.rank(self.decay_linear(p2_ts_rank, int(8.07206)))
        return p1 - p2

    def alpha099(self):
        """Alpha#99: ((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) < rank(correlation(low, volume, 6.28259))) * -1)"""
        p1_inner1 = self.sum((self.high + self.low) / 2, int(19.8975))
        p1_inner2 = self.sum(self.adv60, int(19.8975))
        p1 = self.rank(self.correlation(p1_inner1, p1_inner2, int(8.8136)))
        p2 = self.rank(self.correlation(self.low, self.volume, int(6.28259)))
        return -1 * (p1 < p2).astype(float)
        
    def alpha100(self):
        """Alpha#100: (0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high - close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) - scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))), IndClass.subindustry))) * (volume / adv20))))"""
        p1_num = (self.close - self.low) - (self.high - self.close)
        p1_den = (self.high - self.low).replace(0, 0.0001)
        p1_inner = self.rank((p1_num / p1_den) * self.volume)
        p1 = 1.5 * self.scale(self.indneutralize(self.indneutralize(p1_inner, 'sector'), 'sector'))
        p2_inner1 = self.correlation(self.close, self.rank(self.adv20), 5)
        p2_inner2 = self.rank(self.ts_argmin(self.close, 30))
        p2 = self.scale(self.indneutralize(p2_inner1 - p2_inner2, 'sector'))
        vol_ratio = self.volume / self.adv20.replace(0,1)
        return -1 * (p1 - p2) * vol_ratio

    def alpha101(self):
        """Alpha#101: ((close - open) / ((high - low) + .001))"""
        return (self.close - self.open) / ((self.high - self.low) + 0.001)


    def get_all_alphas(self):
        """Calculates all implemented alpha functions and returns them in a DataFrame."""
        all_alphas = {}
        for i in range(1, 102):
            alpha_name = f'alpha{i:03d}'
            if hasattr(self, alpha_name):
                try:
                    alpha_values = getattr(self, alpha_name)()
                    all_alphas[alpha_name] = alpha_values
                    print(f"Successfully calculated {alpha_name}")
                except Exception as e:
                    print(f"Could not calculate {alpha_name}: {e}")
        return pd.DataFrame(all_alphas)


    # def alpha102(self):
    #     """Calculates the average of some alphas."""
    #     alpha_names = ['alpha001', 'alpha002', 'alpha003', 'alpha004', 'alpha005', 'alpha006', 'alpha008', 'alpha009', 'alpha010', 'alpha011', 'alpha012', 'alpha013', 'alpha014', 'alpha015', 'alpha016', 'alpha017', 'alpha018', 'alpha019', 'alpha020', 'alpha021', 'alpha022', 'alpha024', 'alpha025', 'alpha026', 'alpha028', 'alpha029', 'alpha030', 'alpha031', 'alpha032', 'alpha033', 'alpha034', 'alpha035', 'alpha036', 'alpha037', 'alpha038', 'alpha039', 'alpha040', 'alpha041', 'alpha042', 'alpha043', 'alpha044', 'alpha045', 'alpha046', 'alpha047', 'alpha048', 'alpha049', 'alpha050', 'alpha051', 'alpha052', 'alpha053', 'alpha054', 'alpha055', 'alpha056', 'alpha057', 'alpha058', 'alpha059', 'alpha060', 'alpha066', 'alpha073', 'alpha083', 'alpha084', 'alpha085', 'alpha097']
    #     all_alphas = [getattr(self, name)() for name in alpha_names]
        
    #     # Ensure all alphas have a unique index before concatenation
    #     all_alphas = [alpha.rename(lambda x: f"{name}_{x}", axis=0) for name, alpha in zip(alpha_names, all_alphas)]
        
    #     # Check for non-unique indices and handle them
    #     for alpha in all_alphas:
    #         if alpha.index.duplicated().any():
    #             alpha = alpha[~alpha.index.duplicated(keep='first')]
        
    #     # Calculate the average of the ranks of the alphas
    #     avg_alpha = pd.concat(all_alphas, axis=1).mean(axis=1).rank()
        
    #     return avg_alpha