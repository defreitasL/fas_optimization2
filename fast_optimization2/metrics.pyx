# cython: boundscheck=False
# cython -a -c=-O3 -c=-march=native -c=-ffast-math -c=-funroll-loops
import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double bias(np.ndarray[double, ndim=1] evaluation, np.ndarray[double, ndim=1] simulation):
    """
    Bias objective function
    """
    cdef int n = evaluation.shape[0]
    return np.nansum(evaluation - simulation) / n

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double correlation_coefficient_loss(np.ndarray[double, ndim=1] evaluation, np.ndarray[double, ndim=1] simulation):
    """
    Correlation Coefficient Loss
    """
    cdef int n = evaluation.shape[0]
    cdef double mx = np.nanmean(evaluation)
    cdef double my = np.nanmean(simulation)
    cdef np.ndarray[double, ndim=1] xm = evaluation - mx
    cdef np.ndarray[double, ndim=1] ym = simulation - my
    cdef double r_num = np.nansum(xm * ym)
    cdef double r_den = np.sqrt(np.nansum(np.square(xm)) * np.nansum(np.square(ym)))
    if r_den == 0:
        r_den = 1e-10
    cdef double r = r_num / r_den
    r = np.maximum(np.minimum(r, 1.0), -1.0)
    return 1 - np.square(r)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double mielke_skill_score(np.ndarray[double, ndim=1] evaluation, np.ndarray[double, ndim=1] simulation):
    """
    Mielke index
    if pearson coefficient (r) is zero or positive use kappa=0
    otherwise see Duveiller et al. 2015
    """
    cdef int n = evaluation.shape[0]
    cdef double mx = np.nanmean(evaluation)
    cdef double my = np.nanmean(simulation)
    cdef np.ndarray[double, ndim=1] xm = evaluation - mx
    cdef np.ndarray[double, ndim=1] ym = simulation - my
    cdef np.ndarray[double, ndim=1] diff = (evaluation - simulation) ** 2
    cdef double d1 = np.nansum(diff)
    cdef double d2 = np.nanvar(evaluation) + np.nanvar(simulation) + (np.nanmean(evaluation) - np.nanmean(simulation)) ** 2
    cdef double kappa
    cdef double mss

    if correlation_coefficient_loss(evaluation, simulation) < 0:
        kappa = np.abs(np.nansum(xm * ym)) * 2
        mss = 1 - (d1 / n) / (d2 + kappa)
    else:
        mss = 1 - (d1 / n) / d2

    return mss

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double nashsutcliffe(np.ndarray[double, ndim=1] evaluation, np.ndarray[double, ndim=1] simulation):
    """
    Nash-Sutcliffe objective function
    """
    return 1 - np.nansum((evaluation - simulation) ** 2) / np.nansum((evaluation - np.nanmean(evaluation)) ** 2)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double lognashsutcliffe(np.ndarray[double, ndim=1] evaluation, np.ndarray[double, ndim=1] simulation):
    """
    Log Nash-Sutcliffe objective function
    """
    return 1 - np.nansum((np.log(simulation) - np.log(evaluation)) ** 2) / np.nansum((np.log(evaluation) - np.nanmean(np.log(evaluation))) ** 2)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double pearson(np.ndarray[double, ndim=1] evaluation, np.ndarray[double, ndim=1] simulation):
    """
    Pearson objective function
    """
    return np.corrcoef(evaluation, simulation)[0, 1]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double spearman(np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] y):
    """
    Calculate Spearman's rank correlation coefficient.
    """
    cdef int n = x.shape[0]
    cdef np.ndarray[int, ndim=1] x_rank = np.argsort(np.argsort(x))
    cdef np.ndarray[int, ndim=1] y_rank = np.argsort(np.argsort(y))

    if n == 0:
        return 0.0
    else:
        numerator = 2 * np.nansum(x_rank * y_rank) - n * (n - 1)
        denominator = n * (n - 1) * (n + 1)
        return numerator / denominator

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double agreementindex(np.ndarray[double, ndim=1] evaluation, np.ndarray[double, ndim=1] simulation):
    """
    Agreement Index
    """
    return 1 - (np.nansum((evaluation - simulation) ** 2)) / (np.nansum((np.abs(simulation - np.nanmean(evaluation)) + np.abs(evaluation - np.nanmean(evaluation))) ** 2))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double kge(np.ndarray[double, ndim=1] evaluation, np.ndarray[double, ndim=1] simulation):
    """
    Kling-Gupta Efficiency
    """
    cdef double mu_s = np.nanmean(simulation)
    cdef double mu_o = np.nanmean(evaluation)
    if mu_o == 0:
        mu_o = 1e-10
    cdef double std_s = np.nanstd(simulation)
    cdef double std_o = np.nanstd(evaluation)
    if std_s == 0:
        std_s = 1e-10
    cdef double r = np.corrcoef(simulation, evaluation)[0, 1]
    cdef double beta = mu_s / mu_o
    cdef double alpha = std_s / std_o
    cdef double kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return kge

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double npkge(np.ndarray[double, ndim=1] evaluation, np.ndarray[double, ndim=1] simulation):
    """
    Non parametric Kling-Gupta Efficiency

    Corresponding paper:
    Pool, Vis, and Seibert, 2018 Evaluating model performance: towards a non-parametric variant of the Kling-Gupta efficiency, Hydrological Sciences Journal.

    output:
        kge: Kling-Gupta Efficiency

    author: Nadine Maier and Tobias Houska
    optional_output:
        cc: correlation
        alpha: ratio of the standard deviation
        beta: ratio of the mean
    """
    cdef double cc = spearman(evaluation, simulation)
    cdef double sim_mean = np.nanmean(simulation)
    cdef double eval_mean = np.nanmean(evaluation)
    if eval_mean == 0:
        eval_mean = 1e-10
    if sim_mean == 0:
        sim_mean = 1e-10
    cdef np.ndarray[double, ndim=1] fdc_sim = np.sort(simulation / (sim_mean * len(simulation)))
    cdef np.ndarray[double, ndim=1] fdc_obs = np.sort(evaluation / (eval_mean * len(evaluation)))
    cdef double alpha = 1 - 0.5 * np.nanmean(np.abs(fdc_sim - fdc_obs))
    cdef double beta = sim_mean / eval_mean
    cdef double kge = 1 - np.sqrt((cc - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return kge

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double log_p(np.ndarray[double, ndim=1] evaluation, np.ndarray[double, ndim=1] simulation):
    """
    Logarithmic Probability Distribution
    """
    cdef double scale = np.nanmean(evaluation) / 10
    if scale < 0.01:
        scale = 0.01
    cdef np.ndarray[double, ndim=1] y = (evaluation - simulation) / scale
    cdef np.ndarray[double, ndim=1] normpdf = -(y ** 2) / 2 - np.log(np.sqrt(2 * np.pi))
    return np.nanmean(normpdf)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double covariance(np.ndarray[double, ndim=1] evaluation, np.ndarray[double, ndim=1] simulation):
    """
    Covariance objective function
    """
    cdef double obs_mean = np.nanmean(evaluation)
    cdef double sim_mean = np.nanmean(simulation)
    cdef double covariance = np.nanmean((evaluation - obs_mean) * (simulation - sim_mean))
    return covariance

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double pbias(np.ndarray[double, ndim=1] evaluation, np.ndarray[double, ndim=1] simulation):
    """
    Percent Bias
    """
    return 100 * np.nansum(evaluation - simulation) / np.nansum(evaluation)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double mse(np.ndarray[double, ndim=1] evaluation, np.ndarray[double, ndim=1] simulation):
    """
    Mean Squared Error
    """
    return np.nanmean((evaluation - simulation) ** 2)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double rmse(np.ndarray[double, ndim=1] evaluation, np.ndarray[double, ndim=1] simulation):
    """
    Root Mean Squared Error
    """
    return np.sqrt(np.nanmean((evaluation - simulation) ** 2))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double mae(np.ndarray[double, ndim=1] evaluation, np.ndarray[double, ndim=1] simulation):
    """
    Mean Absolute Error
    """
    return np.nanmean(np.abs(evaluation - simulation))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double rrmse(np.ndarray[double, ndim=1] evaluation, np.ndarray[double, ndim=1] simulation):
    """
    Relative RMSE
    """
    return rmse(evaluation, simulation) / np.nanmean(evaluation)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double rsr(np.ndarray[double, ndim=1] evaluation, np.ndarray[double, ndim=1] simulation):
    """
    RMSE-observations standard deviation ratio
    """
    return rmse(evaluation, simulation) / np.nanstd(evaluation)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double decomposed_mse(np.ndarray[double, ndim=1] evaluation, np.ndarray[double, ndim=1] simulation):
    """
    Decomposed MSE
    """
    cdef double e_std = np.nanstd(evaluation)
    cdef double s_std = np.nanstd(simulation)
    cdef double bias_squared = bias(evaluation, simulation) ** 2
    cdef double sdsd = (e_std - s_std) ** 2
    cdef double lcs = 2 * e_std * s_std * (1 - np.corrcoef(evaluation, simulation)[0, 1])
    return bias_squared + sdsd + lcs

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple backtot():
    """
    Backtot function
    """
    cdef list metrics_name_list = [
        'mss',                      # Max Mielke Skill Score (MSS)
        'nashsutcliffe',            # Max Nash-Sutcliffe Efficiency (NSE)
        'lognashsutcliffe',         # Max log(NSE)
        'pearson',                  # Max Pearson Correlation ($\rho$)
        'spearman',                 # Max Spearman Correlation ($S_{rho}$)
        'agreementindex',           # Max Agreement Index (AI)
        'kge',                      # Max Kling-Gupta Efficiency (KGE)
        'npkge',                    # Max Non-parametric KGE (npKGE)
        'log_p',                    # Max Logarithmic Probability Distribution (LPD)
        'bias',                     # Min Bias (BIAS)
        'pbias',                    # Min Percent Bias (PBIAS)
        'mse',                      # Min Mean Squared Error (MSE)
        'rmse',                     # Min Root Mean Squared Error (RMSE)
        'mae',                      # Min Mean Absolute Error (MAE)
        'rrmse',                    # Min Relative RMSE (RRMSE)
        'rsr',                      # Min RMSE-observations standard deviation ratio (RSR)
        'covariance',               # Min Covariance
        'decomposed_mse',           # Min Decomposed MSE (DMSE)
    ]

    cdef list mask = [
        False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True,
    ]

    return metrics_name_list, mask

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double opt(int index, np.ndarray[double, ndim=1] evaluation, np.ndarray[double, ndim=1] simulation):
    """
    Optimization function
    """
    cdef double out
    if index == 0:
        out = mielke_skill_score(evaluation, simulation)
        if np.isnan(out) or out < 0:
            return 1e-6
        return out
    elif index == 1:
        out = nashsutcliffe(evaluation, simulation)
        if np.isnan(out) or out < 0:
            return 1e-6
        return out
    elif index == 2:
        out = lognashsutcliffe(evaluation, simulation)
        if np.isnan(out) or out < 0:
            return 1e-6
        return out
    elif index == 3:
        out = pearson(evaluation, simulation)
        if np.isnan(out) or out < 0:
            return 1e-6
        return out
    elif index == 4:
        out = spearman(evaluation, simulation)
        if np.isnan(out) or out < 0:
            return 1e-6
        return out
    elif index == 5:
        out = agreementindex(evaluation, simulation)
        if np.isnan(out) or out < 0:
            return 1e-6
        return out
    elif index == 6:
        out = kge(evaluation, simulation)
        if np.isnan(out) or out < 0:
            return 1e-6
        return out
    elif index == 7:
        out = npkge(evaluation, simulation)
        if np.isnan(out) or out < 0:
            return 1e-6
        return out
    elif index == 8:
        out = log_p(evaluation, simulation)
        if np.isnan(out) or out < 0:
            return -999
        return out
    elif index == 9:
        out = bias(evaluation, simulation)
        if np.isnan(out):
            return 1000
        return np.abs(out)
    elif index == 10:
        out = pbias(evaluation, simulation)
        if np.isnan(out):
            return 100
        return out
    elif index == 11:
        out = mse(evaluation, simulation)
        if np.isnan(out):
            return 1000
        return out
    elif index == 12:
        out = rmse(evaluation, simulation)
        if np.isnan(out):
            return 1000
        return out
    elif index == 13:
        out = mae(evaluation, simulation)
        if np.isnan(out):
            return 1000
        return np.abs(out)
    elif index == 14:
        out = rrmse(evaluation, simulation)
        if np.isnan(out):
            return 1000
        return np.abs(out)
    elif index == 15:
        out = rsr(evaluation, simulation)
        if np.isnan(out):
            return 1000
        return np.abs(out)
    elif index == 16:
        out = covariance(evaluation, simulation)
        if np.isnan(out):
            return 1000
        return np.abs(out)
    elif index == 17:
        out = decomposed_mse(evaluation, simulation)
        if np.isnan(out):
            return 1000
        return out
    else:
        raise Warning('Invalid index')
