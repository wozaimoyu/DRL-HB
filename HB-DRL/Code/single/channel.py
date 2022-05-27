
import numpy as np
import math
from random import random
from matplotlib import path
import matplotlib.pyplot as plt


def SteeringGen(theta, Num_ant):
    exponent = 1j * math.pi * theta * np.arange(Num_ant)
    exponent = np.transpose(exponent)
    A = (1 / np.sqrt(Num_ant)) * np.exp(exponent)
    return A.reshape(Num_ant, 1)


class mmWaveChannel:

    def __init__(self):
        '''
        System parameter
        Declaration of factors needed to construct channel (set it as none)
        RL space
        '''
        self.power_db = 0 # transmit power
        self.power_lin = 10**(self.power_db/10)

        self.ULDL = 'UL'

        self.Nu = 1
        self.Nc = 1
        self.Ns_peruser = 1 # 1 data stream per user
        self.Ns = self.Nu * self.Ns_peruser 
        self.Nt = 1  # transmit antennas at BS
        self.Nt_RF = 1
        self.Nr = 16
        self.Nr_RF = 1

        self.r_cell = 1000
        self.dmin = 100

        self.noise = 1

        ## Channel parameters
        self.Np = 1
        self.f_c = 28e9  # Hz
        self.BW = 100e6
        self.ANTgain = 15
        self.NF = 5

        self.AoD = np.empty((self.Nc, self.Np))
        self.AoA = np.empty((self.Nu, self.Nc, self.Np))

    def update(self):
        # TODO: moving users
        return None

        

    def generate(self):

        self._makepoints()
        self._compute_distance()
        self._path_loss_mmWave()
        self.generateangles()

        Nu = self.Nu
        Nc = self.Nc
        Nt = self.Nt  # transmit antennas at BS
        Nr = self.Nr
        self.H = np.zeros((Nr * Nu * Nc, Nt , Nc), dtype=np.complex128)  # a BS <-> a user Nr X Nt

        for c_tx in range(Nc):  # originated from this cell

            for c in range(Nc):
                for u in range(Nu):
                    # idx = Nt * (c * Nu) + u * Nt + np.arange(Nt)
                    start = Nt * (c * Nu) + (u * Nt) + (0)
                    end = Nt * (c * Nu) + (u * Nt) + (Nt - 1)
                    for p in range(self.Np):
                        AR = SteeringGen(self.AoA[u, c, p], self.Nr)
                        AT = SteeringGen(self.AoD[c_tx, p], self.Nt)
                        self.H[:, start:end + 1, c_tx] += self.gain[u, c, c_tx] * np.matmul(AR, AT.T)
                        self.gain[u, c, c_tx] = 0.6 # For simplicity first
                        self.H[:, start:end + 1, c_tx] += self.gain[u, c, c_tx] * AR

                        print(self.gain[0,0,0]**2)
                    # corresponding user-cell indices

    def generateangles(self):
        self.AoD = np.random.uniform(low=-1, high=1, size=(self.Nc, self.Np))
        self.AoA = np.random.uniform(low=-1, high=1, size=(self.Nu, self.Nc, self.Np))

    def _path_loss_mmWave(self):
        D = self.D
        # These are the parameters for f = 28000 MHz.
        PL = 72 + 2.92 * 10 * np.log10(D) + 8.7 * np.random.normal(size=(self.Nu, self.Nc, self.Nc))
        Pnoise = -174 + 10 * np.log10(self.BW) + self.NF
        gamma = 10 ** (-(PL + Pnoise - self.ANTgain) / 10)

        self.gain = np.sqrt(gamma)

    def _compute_distance(self):
        Nu = self.Nu
        Nc = self.Nc
        loc_x = self.loc_x
        loc_y = self.loc_y
        R_max = self.r_cell

        D = np.empty((Nu, Nc, Nc))

        CX = [0.] + np.cos(np.arange(Nc - 1) * math.pi / 3 + math.pi / 6).tolist()  # x coordinates of centers
        CY = [0.] + np.sin(np.arange(Nc - 1) * math.pi / 3 + math.pi / 6).tolist()  # y coordinates of centers

        CX = [x * np.sqrt(3.) * R_max for x in CX]
        CY = [y * np.sqrt(3.) * R_max for y in CY]

        for c in range(Nc):
            D[:, :, c] = np.sqrt((loc_x - CX[c]) ** 2 + (loc_y - CY[c]) ** 2)

        self.D = D

    def _makepoints(self):

        R_max = self.r_cell
        R_min = self.dmin
        Nu = self.Nu
        Nc = self.Nc

        loc_x = np.empty((Nu, Nc))
        loc_y = np.empty((Nu, Nc))

        CX = [0.] + np.cos(np.arange(Nc - 1) * math.pi / 3 + math.pi / 6).tolist()  # x coordinates of centers
        CY = [0.] + np.sin(np.arange(Nc - 1) * math.pi / 3 + math.pi / 6).tolist()  # y coordinates of centers

        CX = [x * np.sqrt(3.) * R_max for x in CX]
        CY = [y * np.sqrt(3.) * R_max for y in CY]

        for c in range(Nc):

            hex_max_x = R_max * np.cos(math.pi * np.arange(6) / 3)
            hex_max_y = R_max * np.sin(math.pi * np.arange(6) / 3)
            hex_min_x = R_min * np.cos(math.pi * np.arange(6) / 3)
            hex_min_y = R_min * np.sin(math.pi * np.arange(6) / 3)

            hex_max = [(hex_max_x[i], hex_max_y[i]) for i in range(6)]
            hex_min = [(hex_min_x[i], hex_min_y[i]) for i in range(6)]

            p_max = path.Path(hex_max)  # square with legs length 1 and bottom left corner at the origin
            p_min = path.Path(hex_min)  # square with legs length 1 and bottom left corner at the origin

            cnt = 0
            while (True):
                P = (R_max - 2 * R_max * random(), R_max - 2 * R_max * random())
                inMAX = p_max.contains_points([P])
                inMIN = p_min.contains_points([P])

                if inMAX == True and inMIN == False:

                    loc_x[cnt][c] = P[0] + CX[c]
                    loc_y[cnt][c] = P[1] + CY[c]
                    cnt += 1

                    if cnt == Nu:
                        break

            #plt.scatter(loc_x[:,c], loc_y[:,c])

        self.loc_x = loc_x
        self.loc_y = loc_y

        #plt.show()


