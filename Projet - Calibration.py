#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 07 08:57:12 2021

@author: william
"""

import csv
import random

import pandas as pd
import math
from scipy import misc
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import scipy
import copy
import random

S = 100
r = 0
max_iter = 1000
p = 500
epsilon = 0.01

df_1an = pd.read_csv("Question 2.1 - Calibration.csv", sep=';')
df_9mois = pd.read_csv("Question 2.2 - Calibration.csv", sep=';')
df_6mois = pd.read_csv("Question 2.3 - Calibration.csv", sep=';')
df_3mois = pd.read_csv("Question 2.4 - Calibration.csv", sep=';')

def Newton_Raphson_implied_volatility(C,S,K,T,r):
    '''
    Newton Raphson method. Compute the implied volatility.
    :param C: Market price
    :param S: Stock
    :param K: Strike
    :param T: Maturity
    :param r: Interest rate
    :return: Implied volatility
    '''
    vol = 0.3
    for i in range(max_iter):
        diff = Black_Scholes_call(S,K,T,r,vol) - C
        if abs(diff) < epsilon:
            break
        vol = vol - diff / Vega_call(S,K,T,r,vol)
    return vol

def Nelder_Mead(f,df,x_start, step=0.1, not_improve_threshold=10e-6, no_improv_break=10, max_iter=0):
    '''
    Nelder Mead method to optimize a given function f
    :param f: function to optimize
    :param df: dataframe
    :param x_strat: starting points
    :param step: step to change the value of the first points
    :param not_improve_threshold: threshold for convergence
    :param no_improv_break: break for convergence
    :param max_iter: number of iterations
    :return: tuple (best parameter array, best score)
    '''

    # initialization for starting ponts
    nb_var = len(x_start[0])
    dim = len(x_start)
    last_best = f(df,x_start)
    not_improv = 0
    res = [[x_start, last_best]]

    for i in range(dim):
        x = copy.copy(x_start)
        for j in range (nb_var):
            x[i][j] = x[i][j] + step
        score = f(df,x)
        res.append([x, score])

    # simplex iter
    iters = 0
    while 1:
        # order f values
        res.sort(key=lambda x: x[1])
        best = res[0][1]

        # break after max_iter
        if max_iter and iters >= max_iter:
            return res[0]
        iters += 1

        if best < last_best - not_improve_threshold:
            not_improv = 0
            last_best = best
        else:
            not_improv += 1

        if not_improv >= no_improv_break:
            return res[0]

        # centroid - find the mid points
        centroid = [0.] * dim
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                centroid[i] += c / (len(res)-1)

        # reflection
        reflection = centroid + (centroid - res[-1][0])
        reflection_score = f(df,reflection)
        if res[0][1] <= reflection_score < res[-2][1]:
            del res[-1]
            res.append([reflection, reflection_score])
            continue

        # expansion
        if reflection_score < res[0][1]:
            expansion = centroid + 2*(centroid - res[-1][0])
            expansion_score = f(df,expansion)
            if expansion_score < reflection_score:
                del res[-1]
                res.append([expansion, expansion_score])
                continue
            else:
                del res[-1]
                res.append([reflection, reflection_score])
                continue

        # contraction
        contraction = centroid + (-0.5)*(centroid - res[-1][0])
        contraction_score = f(df,contraction)
        if contraction_score < res[-1][1]:
            del res[-1]
            res.append([contraction, contraction_score])
            continue

        # reduction
        x1 = res[0][0]
        nres = []
        for tup in res:
            reduction = x1 + (0.5*(tup[0] - x1))
            score = f(df,reduction)
            nres.append([reduction, score])
        res = nres

def Black_Scholes_call(S,K,T,r,vol):
    '''
    Black and Scholes formula for a call.
    :param S: Stock
    :param K: Strike
    :param T: Maturity
    :param r: Interest rate
    :param vol: Volatility
    :return: Call's value.
    '''
    d1 = (math.log(S/K)+(r+vol**2/2.)*T)/(vol*math.sqrt(T))
    d2 = d1 - vol*math.sqrt(T)
    return S*norm.cdf(d1)-K*math.exp(-r*T)*norm.cdf(d2)

def Black_Scholes_call_fractional(S,K,T,r,sigma,H):
    '''
    Black and Scholes formula for a call using fractional brownian motion.
    :param S: Stock
    :param K: Strike
    :param T: Maturity
    :param r: Interest rate
    :param sigma: Volatility
    :param H: Hurst exponent
    :return: Call's value.
    '''
    #d1 = (math.log(S/K)+((r+x[0]**2/2)*(T**(2*x[1]))))/(x[0]*(T**x[1]))
    #d2 = d1 - (x[0]*(T**x[1]))
    #return S*norm.cdf(d1)-K*math.exp(-r*T)*norm.cdf(d2)
    d1 = (math.log(S/K)+((r+sigma**2/2)*(T**(2*H))))/(sigma*(T**H))
    d2 = d1 - (sigma*(T**H))
    return S*norm.cdf(d1)-K*math.exp(-r*T)*norm.cdf(d2)

def Vega_call(S,K,T,r,vol):
    '''
    Vega formula for a call.
    :param S: Stock
    :param K: Strike
    :param T: Maturity
    :param r: Interest rate
    :param vol: Volatility
    :return: Call's vega.
    '''
    d1 = (math.log(S/K)+(r+vol**2/2.)*T)/(vol*math.sqrt(T))
    return S*math.sqrt(T)*norm.pdf(d1)

def get_indexes_min_value(l):
    '''
    Finds index of min avlue in a list
    :param l: list
    :return: Index of the min value
    '''
    min_value = min(l)
    if l.count(min_value) > 1:
        return [i for i, x in enumerate(l) if x == min(l)]
    else:
        return l.index(min(l))

def Reverse_Monte_Carlo(df,n,upper_bound,lower_bound):
    '''
    Minimize a given function thanks to RMC method
    :param df: Dataframe
    :param n: Number of iterations
    :param ub1an: Upper bound
    :param lb1an: Lower bound
    :return: Value that minimize the objective function
    '''
    RMC = []
    vol = []
    for j in range (n):
        vol.append([])
        for i in range (10):
            vol[j].append(np.random.uniform(upper_bound[i],lower_bound[i]))
        RMC.append(Objective_function(df,vol[j]))
    best_value = get_indexes_min_value(RMC)
    return vol[best_value]

def Objective_function(df,x):
    '''
    Objective function to smooth the volatility surface
    :param df: Dataframe
    :param x: List of implied volatilities
    :return: Value of the function
    '''
    K = 95 #first Strike
    sum1 = 0
    sum2 = 0
    H = 0.4
    p = 800
    for i in range (len(x)):
        BS = Black_Scholes_call_fractional(S,K,df['Maturity'][i],0,x[i],H)
        sum1 += (BS-np.array(df['Market Price'])[i])**2
        K += 1
    for i in range (1,len(x)):
        sum2 += (x[i-1] - x[i])**2
    return sum1 + p*sum2

def Objective_function_hurst_one_maturity(df,x):
    '''
    Objective function to find the best implied vol and hurst exponent for one maturity
    :param df: Dataframe
    :param x: Tuple with vol and hurst exponent
    :return: Value of the function
    '''
    sum1 = 0
    sum2 = 0
    sum3 = 0
    p2 = 500
    p3 = 500
    K = 95
    for i in range (len(x[0])):
        BS = Black_Scholes_call_fractional(S,K,df['Maturity'][i],0,x[0][i],x[1][i])
        sum1 += (BS-np.array(df['Market Price'])[i])**2
        K += 1
    for i in range (0,len(x[0])-1):
        sum2 += (x[0][i + 1] - x[0][i]) ** 2
        sum3 += (x[1][i + 1] - x[1][i]) ** 2
    return sum1 + p2*sum2 + p3*sum3

def Objective_function_hurst_all_maturities(df,x):
    '''
    Objective function to find the best implied vol and hurst exponent for all maturities
    :param df: Dataframe
    :param x: Tuple with vol and hurst exponent
    :return: Value of the function
    '''
    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    sum5 = 0
    p2 = 500
    p3 = 1000
    p4 = 800
    p5 = 1000
    T = 1
    dic = Convert_to_dic(df,x)
    for i in range(4):
        K = 95
        for j in range (10):
            BS = Black_Scholes_call_fractional(S,K,T,0,x[0][i],x[1][i])
            sum1 += (BS-np.array(df['Market Price'])[i])**2
            K += 1
        T -= 0.25
    T = 1
    for i in range(4):
        K = 95
        for j in range (9):
            sum2 += (dic[K+1,T][0] - dic[K,T][0]) ** 2
            sum3 += (dic[K+1,T][1] - dic[K,T][1]) ** 2
            K += 1
        T -= 0.25
    K = 95
    for i in range(10):
        T = 1
        for j in range(3):
            sum4 += (dic[K,T-0.25][0] - dic[K,T][0]) ** 2
            sum5 += (dic[K,T-0.25][1] - dic[K,T][1]) ** 2
            T -= 0.25
        K += 1
    return sum1 + p2*sum2 + p3*sum3 + p4*sum4 + p5*sum5

def Smoothing_df(df):
    '''
    Smoothing volatility surface for a given dataframe
    :param df: Dataframe
    :return: list of smoothed volatilities
    '''
    implied_volatility = list(df['ImpliedVol'])
    upper_bound = []
    lower_bound  = []
    for i in range(len(implied_volatility)):
        upper_bound.append(implied_volatility[i] + epsilon)
        lower_bound.append(implied_volatility[i] - epsilon)
    return Reverse_Monte_Carlo(df,5000,upper_bound,lower_bound)

def Compute_implied_volatility(df):
    '''
    Compute and add implied volatility for a given dataframe (maturity)
    :param list_df: Dataframe
    :return: Dataframe with implied vol added
    '''
    implied_vol = []
    for i in range(len(df)):
        implied_vol.append(Newton_Raphson_implied_volatility(df['Market Price'][i], S, df['Strike'][i], df['Maturity'][i], r))
    df = df.assign(ImpliedVol=implied_vol)
    return df

def Compute_smooth_implied_volatility(df):
    '''
    Compute and add smooth implied volatility for a given dataframe (maturity)
    :param list_df: Dataframe
    :return: Dataframe with smooth implied volatility added
    '''
    df = df.assign(SmoothVol=Smoothing_df(df))
    return df

def Compute_price_difference(df):
    '''
    Compute and add price difference between market price and price computed with Black and Scholes formula
    :param df: Dataframe
    :return: Dataframe with price difference added
    '''
    price_difference = []
    for i in range(len(df)):
        price_difference.append(Black_Scholes_call(S,df['Strike'][i],df['Maturity'][i],0,df['SmoothVol'][i]) - df['Market Price'][i])
    df = df.assign(PriceDifference=price_difference)
    return df

def Compute_price_difference_hurst(df):
    '''
    Compute and add price difference between market price and price computed with Black and Scholes formula with hurst exponent
    :param df: Dataframe
    :return: Dataframe with price difference added
    '''
    price_difference = []
    for i in range(len(df)):
        price_difference.append(Black_Scholes_call_fractional(S,df['Strike'][i],df['Maturity'][i],0,df['ImpliedVolWithHurst'][i],df['Hurst'][i]) - df['Market Price'][i])
    df = df.assign(PriceDifference=price_difference)
    return df

def Convert_to_dic(df,x):
    '''
    Method to convert a given dataframe in a dictionary and adding x
    :param df: Dataframe
    :param df: x (sigma and H)
    :return: Dictionary with corresponds to the dataframe
    '''
    dic = {}
    for i in range(len(df)):
        key = ()
        value = 0
        key = (df['Strike'][i],df['Maturity'][i])
        value = (x[0][i], x[1][i])
        dic[key] = value
    return dic

def Monte_Carlo_one_maturity(df):
    '''
    Monte Carlo method on one maturity to find sigma and H
    :param df: Dataframe
    :return: List of the mean value of sigma and H
    '''
    res = []
    for i in range (100):
        vol = list(df_1an['ImpliedVol'])
        hurst = []
        for i in range (10):
            vol[i] = vol[i] + (np.random.normal(0, 1)/10)
            hurst.append(0.5 + (np.random.normal(0, 1) / 10))
        x = np.array([vol, hurst])
        res.append(Nelder_Mead(Objective_function_hurst_one_maturity,df,x)[0])
    monte_carlo_value = []
    for nb_var in range (10):
        mean_vol_list = []
        mean_hurst_list = []
        for tirage in range(100):
            mean_vol_list.append(res[tirage][0][nb_var])
            mean_hurst_list.append(res[tirage][1][nb_var])
        monte_carlo_value.append((np.mean(mean_vol_list),np.mean(mean_hurst_list)))
    return monte_carlo_value

def Monte_Carlo_all_maturities(df):
    '''
    Monte Carlo method on all maturities to find sigma and H
    :param df: Dataframe
    :return: List of the mean value of sigma and H
    '''
    res = []
    for i in range (100):
        vol = list(df['ImpliedVol'])
        hurst = []
        for i in range (40):
            vol[i] = vol[i] + (np.random.normal(0, 1)/10)
            hurst.append(0.5 + (np.random.normal(0, 1) / 10))
        x = np.array([vol, hurst])
        res.append(Nelder_Mead(Objective_function_hurst_all_maturities,df,x)[0])
    monte_carlo_value = []
    for nb_var in range (40):
        mean_vol_list = []
        mean_hurst_list = []
        for tirage in range(100):
            mean_vol_list.append(res[tirage][0][nb_var])
            mean_hurst_list.append(res[tirage][1][nb_var])
        monte_carlo_value.append((np.mean(mean_vol_list),np.mean(mean_hurst_list)))
    return monte_carlo_value

#Question 2 : Implied volatility surface
df_1an = Compute_implied_volatility(df_1an)
df_9mois = Compute_implied_volatility(df_9mois)
df_6mois = Compute_implied_volatility(df_6mois)
df_3mois = Compute_implied_volatility(df_3mois)
# Concatenate all maturities in one dataframe
list_df = [df_1an,df_9mois,df_6mois,df_3mois]
df_all_maturity = pd.concat(list_df)
df_all_maturity = df_all_maturity.reset_index()
df_all_maturity = df_all_maturity.drop(columns='index')
print(df_all_maturity)
#PLOT Q2.1 => PLOT Smooth Volatility for T = 1
'''
fig = plt.figure(figsize=(10, 6))
plt.plot(df_1an['Strike'],df_1an['ImpliedVol'], linewidth=3, label ='Implied Volatility')
plt.title('Implied Volatility for T = 1', fontsize=20)
plt.xlabel('Strike')
plt.ylabel('Volatility')
plt.show()
'''


#PLOT Q2.2 => PLOT Smooth Volatility Surface
'''
fig = plt.figure()
fig = plt.figure(figsize = (12, 8), dpi=80)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(df_all_maturity['Maturity'], df_all_maturity['Strike'], df_all_maturity['ImpliedVol'], cmap=cm.jet, linewidth=0)
cbar = fig.colorbar(surf)
cbar.set_label("Implied Volatility")
fig.set_facecolor('white')
ax.set_facecolor('white')
plt.xticks(np.arange(0, 1.5, 0.5))
plt.yticks(np.arange(94,105,1))
plt.title('Volatility Surface - Rough',fontsize=20)
ax.set_xlabel('Maturity T')
ax.set_ylabel('Strike K')
ax.set_zlabel('Implied Volatility')
plt.show()
'''

#Question 3 : Smooth volatility surface
#Question 3.1 : One maturity T = 1
'''
df_1an = Compute_smooth_implied_volatility(df_1an)
#Question 3.2 : All maturities
df_9mois = Compute_smooth_implied_volatility(df_9mois)
df_6mois = Compute_smooth_implied_volatility(df_6mois)
df_3mois = Compute_smooth_implied_volatility(df_3mois)
# Concatenate all maturities in one dataframe
list_df = [df_1an,df_9mois,df_6mois,df_3mois]
df_all_maturity_smooth = pd.concat(list_df)
df_all_maturity_smooth = df_all_maturity_smooth.reset_index()
df_all_maturity_smooth = df_all_maturity_smooth.drop(columns='index')
# Compute the price difference between market price and price with smooth volatility
df_all_maturity_price = Compute_price_difference(df_all_maturity_smooth)
price_difference = 0
for i in range (len(df_all_maturity_price)):
    price_difference += df_all_maturity_price['PriceDifference'][i]
print(price_difference)
'''
#print(price_difference)

#PLOT Q3.1 => PLOT Smooth Volatility for T = 1
'''
fig = plt.figure(figsize=(10, 6))
plt.plot(df_1an['Strike'],df_1an['SmoothVol'], linewidth=3, label ='Smooth Implied Volatility')
plt.title('Smooth Implied Volatility for T = 1', fontsize=20)
plt.xlabel('Strike')
plt.ylabel('Volatility')
plt.show()
'''


#PLOT Q3.2 => PLOT Smooth Volatility Surface
'''
fig = plt.figure()
fig = plt.figure(figsize = (12, 8), dpi=80)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(df_all_maturity_smooth['Maturity'], df_all_maturity_smooth['Strike'], df_all_maturity_smooth['SmoothVol'], cmap=cm.jet, linewidth=0)
cbar = plt.colorbar(surf)
cbar.set_label("Implied Volatility")
fig.set_facecolor('white')
ax.set_facecolor('white')
plt.xticks(np.arange(0, 1.5, 0.5))
plt.yticks(np.arange(94,105,1))
plt.title('Volatility Surface - Smooth',fontsize=20)
ax.set_xlabel('Maturity T')
ax.set_ylabel('Strike K')
ax.set_zlabel('Implied Volatility')
plt.show()
'''

#Question 4 : Smooth volatility and hurst surface
#Question 4.1 : One maturity T = 1
res = Monte_Carlo_one_maturity(df_1an)
vol = []
hurst = []
for i in range (len(df_1an)):
    vol.append(res[i][0])
    hurst.append(res[i][1])
df_1an = df_1an.assign(ImpliedVolWithHurst=vol)
df_1an = df_1an.assign(Hurst=hurst)

#PLOT Q4.1 => PLOT Smooth Volatility and Hurst for T = 1

fig = plt.figure(figsize=(10, 6))
plt.plot(df_1an['Strike'],df_1an['ImpliedVolWithHurst'], linewidth=3, label ='Implied Volatility')
plt.title('Implied Volatility for T = 1', fontsize=20)
plt.xlabel('Strike')
plt.ylabel('Volatility')
plt.show()

fig = plt.figure(figsize=(10, 6))
plt.plot(df_1an['Strike'],df_1an['Hurst'], linewidth=3, label ='Hurst Exponent')
plt.title('Hurst exponent for T = 1', fontsize=20)
plt.xlabel('Strike')
plt.ylabel('Volatility')
plt.show()

#Question 4.2 : All maturities

res = Monte_Carlo_all_maturities(df_all_maturity)
vol = []
hurst = []
for i in range (len(df_all_maturity)):
    vol.append(res[i][0])
    hurst.append(res[i][1])
df_all_maturity = df_all_maturity.assign(ImpliedVolWithHurst=vol)
df_all_maturity = df_all_maturity.assign(Hurst=hurst)

#PLOT Q4.2 => PLOT Smooth Volatility for all maturities

fig = plt.figure()
fig = plt.figure(figsize = (12, 8), dpi=80)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(df_all_maturity['Maturity'], df_all_maturity['Strike'], df_all_maturity['ImpliedVolWithHurst'], cmap=cm.jet, linewidth=0)
cbar = plt.colorbar(surf)
cbar.set_label("Implied Volatility")
fig.set_facecolor('white')
ax.set_facecolor('white')
plt.xticks(np.arange(0, 1.5, 0.5))
plt.yticks(np.arange(94,105,1))
plt.title('Volatility Surface - Smooth',fontsize=20)
ax.set_xlabel('Maturity T')
ax.set_ylabel('Strike K')
ax.set_zlabel('Implied Volatility')
plt.show()

#PLOT Q4.2 => PLOT Smooth Volatility for all maturities

fig = plt.figure()
fig = plt.figure(figsize = (12, 8), dpi=80)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_trisurf(df_all_maturity['Maturity'], df_all_maturity['Strike'], df_all_maturity['Hurst'], cmap=cm.jet, linewidth=0)
cbar = plt.colorbar(surf)
cbar.set_label("Hurst exponent")
fig.set_facecolor('white')
ax.set_facecolor('white')
plt.xticks(np.arange(0, 1.5, 0.5))
plt.yticks(np.arange(94,105,1))
plt.title('Hurst exponent',fontsize=20)
ax.set_xlabel('Maturity T')
ax.set_ylabel('Strike K')
ax.set_zlabel('Hurst exponent H')
plt.show()

df_all_maturity_price = Compute_price_difference_hurst(df_all_maturity)
price_difference = 0
for i in range (len(df_all_maturity_price)):
    price_difference += df_all_maturity_price['PriceDifference'][i]
print(price_difference)









