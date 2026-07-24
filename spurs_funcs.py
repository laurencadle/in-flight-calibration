import pandas as pd 
import scipy.stats as stats
import os
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

def get_bounds(line, viewing_window = 30, fitting_window = 5):
    lower_bound = line - viewing_window
    upper_bound = line + viewing_window
    fitted_lower= line - fitting_window
    fitted_upper = line + fitting_window
    
    return lower_bound, upper_bound, fitted_lower, fitted_upper

def bin_sorting(N, energy, lower_bound, upper_bound):
    counts, bin_edges = np.histogram(energy, bins=(np.linspace(lower_bound, upper_bound, N))) 
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return counts, bin_edges, bin_width, bin_centers

def slope(d2, d1, y2, y1):
    return (y2 - y1) / (d2 - d1)

def get_masks(fitted_lower, fitted_upper, bin_centers):
    peak_mask = (bin_centers > fitted_lower) & (bin_centers < fitted_upper)
    
    return peak_mask

def gaussian_func(x, mu, sigma, c, d, A):
    gaussian = 1 / (sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2)
    
    return gaussian * (A) + c * x + d

def quadratic_func(x, mu, sigma, c, d, A, e):
    gaussian = 1 / (sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2)
    return gaussian * (A) + c * x ** 2 + d * x + e

def fit_peak(quadratic_func, line, counts, peak_mask, bin_centers, slope_val, fitted_lower, fitted_upper):
    p0_gauss = [line, 2, slope_val, counts[peak_mask].min(), 10000]
    popt, pcov = curve_fit(quadratic_func, bin_centers[peak_mask], counts[peak_mask], p0=p0_gauss, bounds=([fitted_lower, 1, 3*slope_val, 0, 0, -500], [fitted_upper, 10, 0, 1e6, np.inf], 500), maxfev=10000)
    p_err= np.sqrt(np.diag(pcov))
    
    return popt, pcov, p_err

def fit_peak(gaussian_func, line, counts, peak_mask, bin_centers, slope_val, fitted_lower, fitted_upper):
    p0_gauss = [line, 2, slope_val, counts[peak_mask].min(), 10000]
    popt, pcov = curve_fit(gaussian_func, bin_centers[peak_mask], counts[peak_mask], p0=p0_gauss, bounds=([fitted_lower, 1, 3*slope_val, 0, 0], [fitted_upper, 10, 0, 1e6, np.inf]), maxfev=10000)
    p_err= np.sqrt(np.diag(pcov))
    
    return popt, pcov, p_err

def plot_gaussian(line, N, p_err, counts, bin_edges, popt, lower_bound, upper_bound, fitted_lower, fitted_upper, fitting_window, ylim=(0, 1e5)):
    x1 = np.linspace(lower_bound, upper_bound, 1001)
    fitted_model_g = gaussian_func(x1, *popt)
   
    plt.figure(figsize=(10, 8))
    plt.stairs(counts, bin_edges, fill=True, color='turquoise', alpha=0.6, label='Data', linewidth=0.8)
    plt.title(f'Gaussian fit for the {line} keV activation line (bins = {N}, fitting window = {fitting_window} keV)')
    plt.plot(x1, fitted_model_g, color='mediumvioletred', label=f'Gaussian\npeak = {popt[0]:.2f}, std = {popt[1]:.2f}, c = {popt[2]:.2f}, d = {popt[3]:.2f}')
    plt.xlabel('Energy')
    plt.ylabel('Counts')
    plt.xlim(lower_bound, upper_bound)
    plt.ylim(*ylim)
    plt.axvspan(fitted_lower, fitted_upper, alpha=0.12, color='grey', label=f'Fit window ({fitted_lower}–{fitted_upper} keV)')
    plt.errorbar(x=popt[0], y=gaussian_func(popt[0], *popt), xerr=p_err[0], fmt='o', color='red', label=f'Peak error: ±{p_err[0]:.2f} keV')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return x1, fitted_model_g

def plot_quadratic(line, N, p_err, counts, bin_edges, popt, lower_bound, upper_bound, fitted_lower, fitted_upper, fitting_window, ylim=(0, 1e5)):
    x1 = np.linspace(lower_bound, upper_bound, 1001)
    fitted_model_g = quadratic_func(x1, *popt)
   
    plt.figure(figsize=(10, 8))
    plt.stairs(counts, bin_edges, fill=True, color='turquoise', alpha=0.6, label='Data', linewidth=0.8)
    plt.title(f'Quadratic fit for the {line} keV activation line (bins = {N}, fitting window = {fitting_window} keV)')
    plt.plot(x1, fitted_model_g, color='mediumvioletred', label=f'Quadratic\npeak = {popt[0]:.2f}, width = {popt[1]:.2f}, c = {popt[2]:.2f}, d = {popt[3]:.2f}')
    plt.plot([p_err[0]], [0], marker='o', color='red', label=f'Error in peak position: ±{p_err[0]:.2f}')
    plt.xlabel('Energy')
    plt.ylabel('Counts')
    plt.xlim(lower_bound, upper_bound)
    plt.ylim(*ylim)
    plt.axvspan(fitted_lower, fitted_upper, alpha=0.12, color='grey', label=f'Fit window ({fitted_lower}–{fitted_upper} keV)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return x1, fitted_model_g
