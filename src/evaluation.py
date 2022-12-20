import numpy as np
from scipy import signal

def compute_mse(X, Y):
    X = np.float32(X)
    Y = np.float32(Y)
    mse = np.mean((X - Y) ** 2, dtype=np.float64)
    return mse

def compute_psnr(X, Y, data_range):
    mse = compute_mse(X, Y)
    psnr = 10 * np.log10((data_range ** 2) / mse)
    return psnr

def normalize(X, Y, data_range):
    X = X.astype(np.float64) / data_range
    Y = Y.astype(np.float64) / data_range
    return X, Y

def convolve2d(image, kernel):
    result = signal.convolve2d(image, kernel, mode='same', boundary='fill')
    return result

def _ssim_one_channel(X, Y, win_size, data_range):
    X, Y = normalize(X, Y, data_range)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    num = win_size ** 2
    kernel = np.ones([win_size, win_size]) / num
    mean_map_x = convolve2d(X, kernel)
    mean_map_y = convolve2d(Y, kernel)

    mean_map_xx = convolve2d(X * X, kernel)
    mean_map_yy = convolve2d(Y * Y, kernel)
    mean_map_xy = convolve2d(X * Y, kernel)

    cov_norm = num / (num - 1)
    var_x = cov_norm * (mean_map_xx - mean_map_x ** 2)
    var_y = cov_norm * (mean_map_yy - mean_map_y ** 2)
    covar_xy = cov_norm * (mean_map_xy - mean_map_x * mean_map_y)

    A1 = 2 * mean_map_x * mean_map_y + C1
    A2 = 2 * covar_xy + C2
    B1 = mean_map_x ** 2 + mean_map_y ** 2 + C1
    B2 = var_x + var_y + C2

    ssim_map = (A1 * A2) / (B1 * B2)
    mssim = np.mean(ssim_map)
    return mssim