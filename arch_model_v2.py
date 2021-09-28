"""
arch_model_v2 () is an adaptation of the arch_model () function by the author Sheppard (2021) to the needs of this research.

The original function:
https://arch.readthedocs.io/en/latest/univariate/introduction.html#arch.univariate.arch_model

The modified function:
https://github.com/alejandro-cermeno/2021_Market_Timing-Cermeno/blob/main/arch_model_v2.py
"""

from __future__ import annotations

import copy
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)


from arch.univariate.mean import (
    ZeroMean,
    ConstantMean,
    ARX
)

from arch.univariate.volatility import (
    ARCH,
    GARCH,
    EGARCH,
    FIGARCH,
    APARCH
)

from arch.univariate.distribution import (
    GeneralizedError,
    Normal,
    SkewStudent,
    StudentsT,
)


import numpy as np
from pandas import DataFrame, Index
from scipy.optimize import OptimizeResult
from statsmodels.tsa.tsatools import lagmat


def arch_model_v2(
    y: Optional[ArrayLike],
    x: Optional[ArrayLike] = None,
    mean: str = "Constant",
    lags: Optional[Union[int, List[int], NDArray]] = 0,
    vol: str = "Garch",
    p: Union[int, List[int]] = 1,
    o: int = 0,
    q: int = 1,
    power: float = 2.0,
    dist: str = "Normal",
    hold_back: Optional[int] = None,
    rescale: Optional[bool] = None,
) -> HARX:
    """
    Initialization of common ARCH model specifications

    Parameters
    ----------
    y : {ndarray, Series, None}
        The dependent variable
    x : {np.array, DataFrame}, optional
        Exogenous regressors.  Ignored if model does not permit exogenous
        regressors.
    mean : str, optional
        Name of the mean model.  Currently supported options are: 'Constant',
        'Zero', 'LS', 'AR', 'ARX', 'HAR' and  'HARX'
    lags : int or list (int), optional
        Either a scalar integer value indicating lag length or a list of
        integers specifying lag locations.
    vol : str, optional
        Name of the volatility model.  Currently supported options are:
        'GARCH' (default), 'ARCH', 'EGARCH', 'FIARCH' and 'HARCH'
    p : int, optional
        Lag order of the symmetric innovation
    o : int, optional
        Lag order of the asymmetric innovation
    q : int, optional
        Lag order of lagged volatility or equivalent
    power : float, optional
        Power to use with GARCH and related models
    dist : int, optional
        Name of the error distribution.  Currently supported options are:

            * Normal: 'normal', 'gaussian' (default)
            * Students's t: 't', 'studentst'
            * Skewed Student's t: 'skewstudent', 'skewt'
            * Generalized Error Distribution: 'ged', 'generalized error"

    hold_back : int
        Number of observations at the start of the sample to exclude when
        estimating model parameters.  Used when comparing models with different
        lag lengths to estimate on the common sample.
    rescale : bool
        Flag indicating whether to automatically rescale data if the scale
        of the data is likely to produce convergence issues when estimating
        model parameters. If False, the model is estimated on the data without
        transformation.  If True, than y is rescaled and the new scale is
        reported in the estimation results.

    Returns
    -------
    model : ARCHModel
        Configured ARCH model

    Examples
    --------
    >>> import datetime as dt
    >>> import pandas_datareader.data as web
    >>> djia = web.get_data_fred('DJIA')
    >>> returns = 100 * djia['DJIA'].pct_change().dropna()

    A basic GARCH(1,1) with a constant mean can be constructed using only
    the return data

    >>> from arch.univariate import arch_model
    >>> am = arch_model(returns)

    Alternative mean and volatility processes can be directly specified

    >>> am = arch_model(returns, mean='AR', lags=2, vol='harch', p=[1, 5, 22])

    This example demonstrates the construction of a zero mean process
    with a TARCH volatility process and Student t error distribution

    >>> am = arch_model(returns, mean='zero', p=1, o=1, q=1,
    ...                 power=1.0, dist='StudentsT')

    Notes
    -----
    Input that are not relevant for a particular specification, such as `lags`
    when `mean='zero'`, are silently ignored.
    """
    known_mean = ("zero", "constant", "harx", "har", "ar", "arx", "ls")
    known_vol = ("arch", "figarch", "garch", "harch", "constant", "egarch", 
                 "aparch")
    known_dist = (
        "normal",
        "gaussian",
        "studentst",
        "t",
        "skewstudent",
        "skewt",
        "ged",
        "generalized error",
    )
    mean = mean.lower()
    vol = vol.lower()
    dist = dist.lower()
    if mean not in known_mean:
        raise ValueError("Unknown model type in mean")
    if vol.lower() not in known_vol:
        raise ValueError("Unknown model type in vol")
    if dist.lower() not in known_dist:
        raise ValueError("Unknown model type in dist")

    if mean == "harx":
        am = HARX(y, x, lags, hold_back=hold_back, rescale=rescale)
    elif mean == "har":
        am = HARX(y, None, lags, hold_back=hold_back, rescale=rescale)
    elif mean == "arx":
        am = ARX(y, x, lags, hold_back=hold_back, rescale=rescale)
    elif mean == "ar":
        am = ARX(y, None, lags, hold_back=hold_back, rescale=rescale)
    elif mean == "ls":
        am = LS(y, x, hold_back=hold_back, rescale=rescale)
    elif mean == "constant":
        am = ConstantMean(y, hold_back=hold_back, rescale=rescale)
    else:  # mean == "zero"
        am = ZeroMean(y, hold_back=hold_back, rescale=rescale)

    if vol in ("arch", "garch", "figarch", "egarch") and not isinstance(p, int):
        raise TypeError(
            "p must be a scalar int for all volatility processes except HARCH."
        )

    if vol == "constant":
        v: VolatilityProcess = ConstantVariance()
    elif vol == "arch":
        assert isinstance(p, int)
        v = ARCH(p=p)
    elif vol == "figarch":
        assert isinstance(p, int)
        v = FIGARCH(p=p, q=q)
    elif vol == "garch":
        assert isinstance(p, int)
        v = GARCH(p=p, o=o, q=q, power=power)
    elif vol == "egarch":
        assert isinstance(p, int)
        v = EGARCH(p=p, o=o, q=q)
    elif vol == "aparch":          
        assert isinstance(p, int)
        v = APARCH(p=p, o=o, q=q)
    else:  # vol == 'harch'
        v = HARCH(lags=p)

    if dist in ("skewstudent", "skewt"):
        d: Distribution = SkewStudent()
    elif dist in ("studentst", "t"):
        d = StudentsT()
    elif dist in ("ged", "generalized error"):
        d = GeneralizedError()
    else:  # ('gaussian', 'normal')
        d = Normal()

    am.volatility = v
    am.distribution = d

    return am
