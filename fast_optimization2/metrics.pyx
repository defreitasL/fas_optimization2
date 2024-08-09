# cython: boundscheck=False
# cython -a -c=-O3 -c=-march=native -c=-ffast-math -c=-funroll-loops
from libc.stdlib cimport malloc, free, rand, srand
from libc.math cimport sqrt, exp, log, fabs, isnan, pow, pi
from time import time as time_now
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
    cdef double total_diff = 0.0
    cdef int i

    for i in range(n):
        if not isnan(evaluation[i]) and not isnan(simulation[i]):
            total_diff += evaluation[i] - simulation[i]

    return total_diff / n

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
    cdef double r_num = 0.0
    cdef double r_den_x = 0.0
    cdef double r_den_y = 0.0
    cdef double xm, ym
    cdef int i

    for i in range(n):
        if not isnan(evaluation[i]) and not isnan(simulation[i]):
            xm = evaluation[i] - mx
            ym = simulation[i] - my
            r_num += xm * ym
            r_den_x += xm * xm
            r_den_y += ym * ym

    r_den = sqrt(r_den_x * r_den_y)
    if r_den == 0:
        r_den = 1e-10

    cdef double r = r_num / r_den
    r = max(min(r, 1.0), -1.0)

    return 1 - r * r

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
    cdef double d1 = 0.0
    cdef double d2 = 0.0
    cdef double kappa = 0.0
    cdef double mss = 0.0
    cdef int i

    for i in range(n):
        if not isnan(evaluation[i]) and not isnan(simulation[i]):
            d1 += pow(evaluation[i] - simulation[i], 2)
            d2 += pow(evaluation[i] - mx, 2) + pow(simulation[i] - my, 2)

    if correlation_coefficient_loss(evaluation, simulation) < 0:
        for i in range(n):
            if not isnan(evaluation[i]) and not isnan(simulation[i]):
                kappa += fabs((evaluation[i] - mx) * (simulation[i] - my))
        kappa *= 2
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
    cdef int n = evaluation.shape[0]
    cdef double mean_eval = np.nanmean(evaluation)
    cdef double num = 0.0
    cdef double den = 0.0
    cdef int i

    for i in range(n):
        if not isnan(evaluation[i]) and not isnan(simulation[i]):
            num += pow(evaluation[i] - simulation[i], 2)
            den += pow(evaluation[i] - mean_eval, 2)

    return 1 - num / den

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double lognashsutcliffe(np.ndarray[double, ndim=1] evaluation, np.ndarray[double, ndim=1] simulation):
    """
    Log Nash-Sutcliffe objective function
    """
    cdef int n = evaluation.shape[0]
    cdef double mean_log_eval = np.nanmean(np.log(evaluation))
    cdef double num = 0.0
    cdef double den = 0.0
    cdef int i

    for i in range(n):
        if not isnan(evaluation[i]) and not isnan(simulation[i]):
            num += pow(log(simulation[i]) - log(evaluation[i]), 2)
            den += pow(log(evaluation[i]) - mean_log_eval, 2)

    return 1 - num / den

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
    cdef double numerator = 0.0
    cdef double denominator = 0.0
    cdef int i

    if n == 0:
        return 0.0

    for i in range(n):
        numerator += (x_rank[i] - y_rank[i]) * (x_rank[i] - y_rank[i])

    denominator = n * (n * n - 1)

    return 1 - (6 * numerator / denominator)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double agreementindex(np.ndarray[double, ndim=1] evaluation, np.ndarray[double, ndim=1] simulation):
    """
    Agreement Index
    """
    cdef double mean_eval = np.nanmean(evaluation)
    cdef double num = 0.0
    cdef double den = 0.0
    cdef int n = evaluation.shape[0]
    cdef int i

    for i in range(n):
        if not isnan(evaluation[i]) and not isnan(simulation[i]):
            num += pow(evaluation[i] - simulation[i], 2)
            den += pow(fabs(simulation[i] - mean_eval) + fabs(evaluation[i] - mean_eval), 2)

    return 1 - num / den

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
    cdef double kge = 1 - sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
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
    cdef double kge = 1 - sqrt((cc - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
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
    cdef double total = 0.0
    cdef double y, normpdf
    cdef int n = evaluation.shape[0]
    cdef int i

    for i in range(n):
        if not isnan(evaluation[i]) and not isnan(simulation[i]):
            y = (evaluation[i] - simulation[i]) / scale
            normpdf = -(y * y) / 2 - log(sqrt(2 * pi))
            total += normpdf

    return total / n

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double covariance(np.ndarray[double, ndim=1] evaluation, np.ndarray[double, ndim=1] simulation):
    """
    Covariance objective function
    """
    cdef double obs_mean = np.nanmean(evaluation)
    cdef double sim_mean = np.nanmean(simulation)
    cdef double total_covariance = 0.0
    cdef int n = evaluation.shape[0]
    cdef int i

    for i in range(n):
        if not isnan(evaluation[i]) and not isnan(simulation[i]):
            total_covariance += (evaluation[i] - obs_mean) * (simulation[i] - sim_mean)

    return total_covariance / n

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double pbias(np.ndarray[double, ndim=1] evaluation, np.ndarray[double, ndim=1] simulation):
    """
    Percent Bias
    """
    cdef double total_eval = 0.0
    cdef double total_diff = 0.0
    cdef int n = evaluation.shape[0]
    cdef int i

    for i in range(n):
        if not isnan(evaluation[i]) and not isnan(simulation[i]):
            total_eval += evaluation[i]
            total_diff += evaluation[i] - simulation[i]

    return 100 * total_diff / total_eval

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double mse(np.ndarray[double, ndim=1] evaluation, np.ndarray[double, ndim=1] simulation):
    """
    Mean Squared Error
    """
    cdef double total = 0.0
    cdef int n = evaluation.shape[0]
    cdef int i

    for i in range(n):
        if not isnan(evaluation[i]) and not isnan(simulation[i]):
            total += pow(evaluation[i] - simulation[i], 2)

    return total / n

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double rmse(np.ndarray[double, ndim=1] evaluation, np.ndarray[double, ndim=1] simulation):
    """
    Root Mean Squared Error
    """
    return sqrt(mse(evaluation, simulation))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef double mae(np.ndarray[double, ndim=1] evaluation, np.ndarray[double, ndim=1] simulation):
    """
    Mean Absolute Error
    """
    cdef double total = 0.0
    cdef int n = evaluation.shape[0]
    cdef int i

    for i in range(n):
        if not isnan(evaluation[i]) and not isnan(simulation[i]):
            total += fabs(evaluation[i] - simulation[i])

    return total / n

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
    cdef double bias_squared = pow(bias(evaluation, simulation), 2)
    cdef double sdsd = pow(e_std - s_std, 2)
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
        if isnan(out) or out < 0:
            return 1e-6
        return out
    elif index == 1:
        out = nashsutcliffe(evaluation, simulation)
        if isnan(out) or out < 0:
            return 1e-6
        return out
    elif index == 2:
        out = lognashsutcliffe(evaluation, simulation)
        if isnan(out) or out < 0:
            return 1e-6
        return out
    elif index == 3:
        out = pearson(evaluation, simulation)
        if isnan(out) or out < 0:
            return 1e-6
        return out
    elif index == 4:
        out = spearman(evaluation, simulation)
        if isnan(out) or out < 0:
            return 1e-6
        return out
    elif index == 5:
        out = agreementindex(evaluation, simulation)
        if isnan(out) or out < 0:
            return 1e-6
        return out
    elif index == 6:
        out = kge(evaluation, simulation)
        if isnan(out) or out < 0:
            return 1e-6
        return out
    elif index == 7:
        out = npkge(evaluation, simulation)
        if isnan(out) or out < 0:
            return 1e-6
        return out
    elif index == 8:
        out = log_p(evaluation, simulation)
        if isnan(out) or out < 0:
            return -999
        return out
    elif index == 9:
        out = bias(evaluation, simulation)
        if isnan(out):
            return 1000
        return fabs(out)
    elif index == 10:
        out = pbias(evaluation, simulation)
        if isnan(out):
            return 100
        return out
    elif index == 11:
        out = mse(evaluation, simulation)
        if isnan(out):
            return 1000
        return out
    elif index == 12:
        out = rmse(evaluation, simulation)
        if isnan(out):
            return 1000
        return out
    elif index == 13:
        out = mae(evaluation, simulation)
        if isnan(out):
            return 1000
        return fabs(out)
    elif index == 14:
        out = rrmse(evaluation, simulation)
        if isnan(out):
            return 1000
        return fabs(out)
    elif index == 15:
        out = rsr(evaluation, simulation)
        if isnan(out):
            return 1000
        return fabs(out)
    elif index == 16:
        out = covariance(evaluation, simulation)
        if isnan(out):
            return 1000
        return fabs(out)
    elif index == 17:
        out = decomposed_mse(evaluation, simulation)
        if isnan(out):
            return 1000
        return out
    else:
        raise Warning('Invalid index')
