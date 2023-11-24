import numpy as np

class Lagrange:
    def __init__(self, data, linspace_start, linspace_end, linspace_step) -> None:
        self.data = data
        self.lin_x = np.linspace(linspace_start, linspace_end ,linspace_step)
        self.l_poly = self.lagrange_poly()
        self.funkcja_done = self.wzor_lagrange()

    def lagrange_poly(self):
        l_return = np.ones((self.data.shape[0], self.lin_x.shape[0]))
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[0]):
                if j!=i:
                    l_return[i,:] *= (self.lin_x - self.data[j,0]) / (self.data[i,0] - self.data[j,0])

        return l_return

    def wzor_lagrange(self):
        p_return = np.zeros(self.lin_x.shape[0])
        for n in range(self.data.shape[0]):
            p_return += self.l_poly[n, :] * self.data[n, 1]
        
        return p_return

class Sklejana1st:
    def __init__(self, data, linspace_start, linspace_end, linspace_limit) -> None:
        self.data  = data
        self.start = linspace_start
        self.end   = linspace_end
        self.limit = linspace_limit
        self.lin_x = np.linspace(linspace_start, linspace_end, linspace_limit)
        self.funkcja_done = self.S_i()

    def a_i(self):
        a_return = np.zeros(self.data.shape[0])

        for i in range(self.data.shape[0] - 1):
            a_return[i] = (self.data[i+1, 1] - self.data[i, 1]) / (self.data[i+1, 0] - self.data[i, 0])
        
        return a_return

    def S_i(self):
        Si = np.zeros(self.lin_x.shape[0])
        for i in range(self.data.shape[0] - 1):
            Si += ((self.data[i+1, 1] - self.data[i, 1])/(self.data[i+1, 0] - self.data[i, 0]) * (self.lin_x - self.data[i, 0]) + self.data[i, 1]) * (self.lin_x < self.data[i+1, 0]) * (self.lin_x >= self.data[i, 0]) 

        Si += ((self.data[1, 1] - self.data[0, 1])/(self.data[1, 0] - self.data[0, 0]) * (self.lin_x - self.data[0, 0]) + self.data[0, 1]) * (self.lin_x < self.data[0, 0]) * (self.lin_x >= self.start)
        Si += ((self.data[-1, 1] - self.data[-2, 1])/(self.data[-1, 0] - self.data[-2, 0]) * (self.lin_x - self.data[-2, 0]) + self.data[-2, 1]) * (self.lin_x < self.end) * (self.lin_x >= self.data[-1, 0])
        Si[-1] += ((self.data[-1, 1] - self.data[-2, 1])/(self.data[-1, 0] - self.data[-2, 0]) * (self.lin_x[-1] - self.data[-2, 0]) + self.data[-2, 1])

        return Si


class Sklejana3st:
    def __init__(self, data, linspace_start, linspace_end, linspace_step) -> None:
        self.data = data
        self.lin_end = linspace_end
        self.lin_start = linspace_start
        self.lin_x = np.linspace(linspace_start, linspace_end, linspace_step)
        self.h_i = self.h_and_b()['h']
        self.b_i = self.h_and_b()['b']
        self.u1 = 2*(self.h_i[0] + self.h_i[1])
        self.v1 = self.b_i[1] - self.b_i[0]
        self.n = data.shape[0] - 1
        self.z = self.z_i()
        self.funkcja_done = self.fcja_sklejana_done()

    def u_i(self):
        u_return = np.zeros(self.n - 1)
        u_return[0] = self.u1

        for i in range(1, self.n - 1):
            u_return[i] = 2*(self.h_i[i] + self.h_i[i+1]) - (self.h_i[i]**2)/u_return[i-1]

        return u_return
    
    def v_i(self):
        u_values = self.u_i()
        v_return = np.zeros(self.n - 1)
        v_return[0] = self.v1

        for i in range(1, self.n - 1):
            v_return[i] = self.b_i[i + 1] - self.b_i[i] - (self.h_i[i] * v_return[i-1])/u_values[i-1]

        return v_return

    def z_i(self):
        z_return = np.zeros(self.n + 1)
        v_values = self.v_i()
        u_values = self.u_i()

        for i in range(self.n - 1, 0, -1):
            z_return[i] = (v_values[i-1] - (self.h_i[i] * z_return[i+1]))/u_values[i-1]

        return z_return
    
    def h_and_b(self) -> dict:
        hb_return = {
            'h': np.zeros(self.data.shape[0] - 1),
            'b': np.zeros(self.data.shape[0] - 1)
        }

        for i in range(hb_return['h'].shape[0]):
            hb_return['h'][i] = self.data[i+1, 0] - self.data[i, 0]
            hb_return['b'][i] = (6/hb_return['h'][i]) * (self.data[i+1, 1] - self.data[i, 1])

        return hb_return
    
    def fcja_sklejana_done(self):
        a = np.zeros(self.data.shape[0] - 1)
        b = np.zeros(self.data.shape[0] - 1)
        c = np.zeros(self.data.shape[0] - 1)
        sklej = np.zeros(self.lin_x.shape[0])

        for i in range(self.data.shape[0] - 1):
            a[i] = (1/(6 * self.h_i[i])) * (self.z[i+1] - self.z[i])
            b[i] = self.z[i]/2
            c[i] = ((-1 * self.h_i[i])/6) * (self.z[i+1] + (2 * self.z[i])) + 1/self.h_i[i] * (self.data[i+1, 1] - self.data[i, 1])

        for i in range(self.data.shape[0] - 1):
            sklej += (self.data[i, 1] + (self.lin_x - self.data[i, 0]) * (c[i] + (self.lin_x - self.data[i, 0]) * (b[i] + (self.lin_x - self.data[i, 0]) * a[i]))) * ((self.lin_x < self.data[i + 1, 0]) * (self.lin_x >= self.data[i, 0]))

        sklej += (self.data[0, 1] + (self.lin_x - self.data[0, 0]) * (c[0] + (self.lin_x - self.data[0, 0]) * (b[0] + (self.lin_x - self.data[0, 0]) * a[0]))) * ((self.lin_x < self.data[0, 0]) * (self.lin_x >= self.lin_start))
        sklej += (self.data[-1, 1] + (self.lin_x - self.data[-1, 0]) * (c[-1] + (self.lin_x - self.data[-1, 0]) * (b[-1] + (self.lin_x - self.data[-1, 0]) * a[-1]))) * ((self.lin_x < self.lin_end) * (self.lin_x >= self.data[-1, 0]))
        sklej[-1] += (self.data[-1, 1] + (self.lin_x[-1] - self.data[-1, 0]) * (c[-1] + (self.lin_x[-1] - self.data[-1, 0]) * (b[-1] + (self.lin_x[-1] - self.data[-1, 0]) * a[-1])))

        return sklej
