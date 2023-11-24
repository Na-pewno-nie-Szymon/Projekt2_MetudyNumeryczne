'''
Zad.2. Na jednym rysunku przedstawić wykresy funkcji sklejanej sześciennej
z poprzedniego zadania, funkcji sklejanej stopnia 1 (z ostatnich ćwiczeń) oraz
wielomianu interpolującego otrzymanego ze wzoru Lagrange’a dla danych z
Zad.1. Na wykresie powinna znaleźć się legenda z oznaczeniem każdej z
funkcji.
'''

import matplotlib.pyplot as plt
import interpol as pl
import numpy as np
import scipy.interpolate as scp

data = np.array([[1., 3.],[2., 1.],[3.5, 4.],[5., 0.],[6., .5],[9., -2.],[9.5, -3.]])
lin_start = -2
lin_end = 12
lin_step = 1000
x = np.linspace(lin_start, lin_end, lin_step)



lagrange = pl.Lagrange(data, lin_start, lin_end, lin_step).funkcja_done
skl_3st = pl.Sklejana3st(data, lin_start, lin_end, lin_step).funkcja_done
skl_1st = pl.Sklejana1st(data, lin_start, lin_end, lin_step).funkcja_done

plt.plot(x, lagrange, label='Lagrange')
plt.plot(x, skl_1st, label='Sklejana 1st')
plt.plot(x, skl_3st, label='Sklejana 3st')
plt.plot(data[:, 0], data[:, 1], marker="*")
plt.legend()

plt.show()

poly = scp.CubicSpline(data[:, 0], data[:, 1])
poly = poly(x)


plt.plot(x, skl_3st, label='Sklejana 3st')
plt.plot(x, poly, label='CubicSpline')
plt.legend()

plt.show()
