#!/usr/bin/env python3

import random
import numpy as np
import matplotlib.pyplot as plt
import pickle


class Fourier:
    """."""

    def __init__(self, cos_harms=None, cos_coeff=None, sin_harms=None, sin_coeff=None):
        self.cos_harms = list() if cos_harms is None else cos_harms
        self.cos_coeff = list() if cos_coeff is None else cos_coeff
        self.sin_harms = list() if sin_harms is None else sin_harms
        self.sin_coeff = list() if sin_coeff is None else sin_coeff

    def calc_func(self, nrpts):
        chs, shs = self.cos_harms, self.sin_harms
        cn, sn = self.cos_coeff, self.sin_coeff
        t = np.linspace(0, 1, nrpts)
        p = np.zeros(t.shape)
        for i in range(len(chs)):
            n = chs[i]
            p += cn[i] * np.cos(n * 2 * np.pi * t)
        for i in range(len(shs)):
            n = shs[i]
            p += sn[i] * np.sin(n * 2 * np.pi * t)
        return p

    def calc_matrix(self, fvals):
        cos_harms = self.cos_harms
        sin_harms = self.sin_harms
        t = np.linspace(0, 1, len(fvals))
        csize, ssize = len(cos_harms), len(sin_harms)

        mat = np.zeros((csize + ssize, csize + ssize))
        vec = np.zeros(csize + ssize)

        for i in range(len(cos_harms)):
            n1 = cos_harms[i]
            f1 = np.cos(2*np.pi*n1*t)
            vec[i] = np.inner(fvals, f1)
            for j in range(len(cos_harms)):
                n2 = cos_harms[j]
                f2 = np.cos(2*np.pi*n2*t)
                mat[i, j] = np.inner(f1, f2)
                # plt.plot(f1, '.-')
                # plt.plot(f2, '.-')
                # plt.title(f'n1 = {n1}, n2 = {n2}')
                # plt.show()
            for j in range(len(sin_harms)):
                n2 = sin_harms[j]
                f2 = np.sin(2*np.pi*n2*t)
                mat[i, csize + j] = np.inner(f1, f2)

        for i in range(len(sin_harms)):
            n1 = sin_harms[i]
            f1 = np.sin(2*np.pi*n1*t)
            vec[csize + i] = np.inner(fvals, f1)
            for j in range(len(cos_harms)):
                n2 = cos_harms[j]
                f2 = np.cos(2*np.pi*n2*t)
                mat[csize + i, j] = np.inner(f1, f2)
            for j in range(len(sin_harms)):
                n2 = sin_harms[j]
                f2 = np.sin(2*np.pi*n2*t)
                mat[csize + i, csize + j] = np.inner(f1, f2)

        return mat, vec

    def calc_coeffs(self, fevals):
        mat, vec = self.calc_matrix(fevals)
        coeffs = np.linalg.solve(mat, vec)
        self.cos_coeff = coeffs[:len(self.cos_harms)]
        self.sin_coeff = coeffs[len(self.cos_harms):]


class GParams:
    plot_lines = False
    delta_coeff = 1.0
    plot_lim_factor = 1
    iters = 0
    iters_failured = 0
    iters_failures_for_decrease = 1000
    decrease_factor = 0.95
    iters_improved = list()
    plot0_type = '.-'
    plot1_type = '-'


def create_figure0():
    nrpts = 10
    x1 = np.linspace(-1, 1, nrpts)
    y1 = np.linspace(-1, -1, nrpts)
    x2 = np.linspace(1, 1, nrpts)
    y2 = np.linspace(-1, 1, nrpts)
    x3 = np.linspace(1, -1, nrpts)
    y3 = np.linspace(1, 1, nrpts)
    x4 = np.linspace(-1, -1, nrpts)
    y4 = np.linspace(1, -1, nrpts)
    x = np.hstack((x1, x2, x3, x4))
    y = np.hstack((y1, y2, y3, y4))
    return x, y


def create_figure1():
    x = [-4,-3,-2,-1,0,1,2,3,4,4,4,3,2,1,1,1,1,1,1,1,1,1,1,1,1,0,-1,-2,-3,-3,-3,-2,-1,-1,-1,-1,-1,-1,-1,-1,-2,-3,-4,-4,-4]
    y = [-6,-6,-6,-6,-6,-6,-6,-6,-6,-5,-4,-4,-4,-4,-3,-2,-1,0,1,2,3,4,5,6,7,7,7,6,5,4,3,3,3,2,1,0,-1,-2,-3,-4,-4,-4,-4,-5,-6]
    x = np.array(x)
    y = np.array(y)
    return x, y


def create_figure2():
    t = np.linspace(0, 1, 100)
    x = np.cos(2*np.pi*t)
    y = np.sin(2*np.pi*t)
    return x, y


def create_figure3():
    x = [1, 1, 3, 3, 1, 1, -1, -1, -3, -3, -1, -1, 1] 
    y = [-3, -1, -1, 1, 1, 3, 3, 1, 1, -1, -1, -3, -3]
    x = np.array(x)
    y = np.array(y)
    return x,y 


def create_figure4():
    x = [0, -2, -2, -3, -4, -4, -6, -6, -4, -3, -2, 0, 0] + [2, 2, 8, 8, 4, 4, 8, 8, 4, 4, 8, 8, 2, 8] + [10, 10, 12, 12, 16, 16, 10]
    y = [0, 0, 8, 7, 8, 0, 0, 10, 10, 9, 10, 10, 0] + [0, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 0, 0, 0] + [0, 10, 10, 2, 2, 0, 0]
    x = np.array(x)
    y = np.array(y)
    return x, y

def create_figure5():

    d = list()
    d.append((np.linspace(0, 0, 10), np.linspace(0, 10, 10)))
    d.append((np.linspace(0, 2, 10), np.linspace(10, 10, 10)))
    d.append((np.linspace(2, 3, 10), np.linspace(10, 9, 10)))
    d.append((np.linspace(3, 4, 10), np.linspace(9, 10, 10)))
    d.append((np.linspace(4, 6, 10), np.linspace(10, 10, 10)))
    d.append((np.linspace(6, 6, 10), np.linspace(10, 0, 10)))
    d.append((np.linspace(6, 4, 10), np.linspace(0, 0, 10)))
    d.append((np.linspace(4, 4, 10), np.linspace(0, 8, 10)))
    d.append((np.linspace(4, 3, 10), np.linspace(8, 7, 10)))
    d.append((np.linspace(3, 2, 10), np.linspace(7, 8, 10)))
    d.append((np.linspace(2, 2, 10), np.linspace(8, 0, 10)))
    d.append((np.linspace(2, 0, 10), np.linspace(0, 0, 10)))

    x, y = list(), list()
    for d_ in d:
        x_, y_ = d_
        x += list(x_)
        y += list(y_)
    x = np.array(x)
    y = np.array(y)
    return x, y


def calculate_figure(x_cn, x_sn, y_cn, y_sn, nrpts):
    t = np.linspace(0, 1, nrpts)
    x = np.zeros(t.shape)
    y = np.zeros(t.shape)
    for n in range(len(x_cn)):
        x += x_cn[n] * np.cos(n * 2 * np.pi * t) + x_sn[n] * np.sin(n * 2 * np.pi * t)
        y += y_cn[n] * np.cos(n * 2 * np.pi * t) + y_sn[n] * np.sin(n * 2 * np.pi * t)
    return x, y


def calc_residue(x0, y0, x1, y1):
    r = np.sum(((x1 - x0)**2 + (y1 - y0)**2)**0.5) / len(x0)
    return r


def new_fourier(x_cn, x_sn, y_cn, y_sn):
    x_cn = x_cn.copy()
    x_sn = x_sn.copy()
    y_cn = y_cn.copy()
    y_sn = y_sn.copy()
    for n in range(len(x_cn)):
        x_cn[n] += 2*(random.random() - 0.5) * GParams.delta_coeff
        x_sn[n] += 2*(random.random() - 0.5) * GParams.delta_coeff
        y_cn[n] += 2*(random.random() - 0.5) * GParams.delta_coeff
        y_sn[n] += 2*(random.random() - 0.5) * GParams.delta_coeff
    return x_cn, x_sn, y_cn, y_sn


def plot_fit(x0, y0, x1, y1, res):
    plt.plot(x0, y0, GParams.plot0_type)
    plt.plot(x1, y1, GParams.plot1_type)
    xc, yc = np.mean(x0), np.mean(y0)
    sx, sy = np.max(x0) - np.min(x0), np.max(y0) - np.min(y0)
    if GParams.plot_lines:
        for n in range(x0.size - 1):
            lx = [x0[n], x1[n]]
            ly = [y0[n], y1[n]]
            plt.plot(lx, ly, color='C2')
    lf = GParams.plot_lim_factor
    plt.xlim([xc - sx*lf, xc + sx*lf])
    plt.ylim([yc - sy*lf, yc + sy*lf])
    plt.title(f'iter: {GParams.iters}, resíduo: {res:.6f}')
    plt.gca().set_aspect('equal')
    plt.show()


def run_search(func_figure):

    x0, y0 = func_figure()

    theta = 0*np.pi*1.25
    c = np.cos(theta)
    s = np.sin(theta)
    x_cn = [0,  c, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    x_sn = [0, -s, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
    y_cn = [0,  c, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    y_sn = [0,  s, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # x_cn = [0, c]
    # x_sn = [0, -s]
    # y_cn = [0, c]
    # y_sn = [0, s]

    x1, y1 = calculate_figure(x_cn, x_sn, y_cn, y_sn, x0.size)
    res = calc_residue(x0, y0, x1, y1)
    plot_fit(x0, y0, x1, y1, res)

    GParams.iters = 0
    GParams.iters_failured = 0
    GParams.iters_improved.append(0)

    while True:

        x_cn1, x_sn1, y_cn1, y_sn1 = new_fourier(x_cn, x_sn, y_cn, y_sn)
        x1, y1 = calculate_figure(x_cn1, x_sn1, y_cn1, y_sn1, x0.size)
        res1 = calc_residue(x0, y0, x1, y1)

        if res1 < res:
            x_cn, x_sn, y_cn, y_sn = x_cn1, x_sn1, y_cn1, y_sn1
            res = res1
            GParams.iters += 1
            GParams.iters_failured = 0
            plot_fit(x0, y0, x1, y1, res)
            GParams.iters_improved.append(GParams.iters)
        else:
            GParams.iters_failured += 1
            GParams.iters += 1

        if GParams.iters % 100 == 0:
            print(f'iteration: {GParams.iters}')

        if GParams.iters_failured >= GParams.iters_failures_for_decrease:
            GParams.iters_failured = 0
            GParams.delta_coeff *= GParams.decrease_factor
            print(f'decreased delta: {GParams.delta_coeff}')


def run_fit(func_figure):

    x0, y0 = func_figure()

    cos_harms = [i for i in range(0, 27)]
    sin_harms = [i for i in range(1, 27)]
    
    x_fourier = Fourier(cos_harms, None, sin_harms, None)
    x_fourier.calc_coeffs(x0)
    x1 = x_fourier.calc_func(10*len(x0))
    
    y_fourier = Fourier(cos_harms, None, sin_harms, None)
    y_fourier.calc_coeffs(y0)
    y1 = y_fourier.calc_func(10*len(y0))

    data = dict(x_fourier=x_fourier, y_fourier=y_fourier)
    with open('mel.pkl', 'wb') as f:
        pickle.dump(data, f)

    # res = calc_residue(x0, y0, x1, y1)
    plot_fit(x0, y0, x1, y1, res=0)



# x0, y0 = create_figure5()
# plot_fit(x0, y0, x0, y0, 0)

# run_search(create_figure2)
run_fit(create_figure5)
