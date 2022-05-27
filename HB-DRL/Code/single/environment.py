from channel import SteeringGen
from channel import mmWaveChannel
import numpy as np
import torch




class radio_env:
    def __init__(self):
        self.SP = mmWaveChannel()
        self.SP.generate()

        self.state = 0

        if self.SP.ULDL == 'UL':

            # Digital Precoder at TX
            self.F_BB = np.random.uniform(size=(self.SP.Nt_RF, self.SP.Ns, self.SP.Nu, self.SP.Nc)) \
                     + 1j*np.random.random(size=(self.SP.Nt_RF, self.SP.Ns, self.SP.Nu, self.SP.Nc))  # a BS <-> a user Nr X Nt

            # Digital Combiner at RX
            self.W_BB = np.random.uniform(size=(self.SP.Nr_RF, self.SP.Ns_peruser, self.SP.Nu, self.SP.Nc)) \
                     + 1j*np.random.uniform(size=(self.SP.Nr_RF, self.SP.Ns_peruser, self.SP.Nu, self.SP.Nc))


            if self.SP.Nt_RF == 1:
                self.F_BB[0, 0, 0, 0] = 1.
            if self.SP.Nr_RF == 1:
                self.W_BB[0,0,0,0] = 1.

            self.F_RF = np.zeros((self.SP.Nt, self.SP.Nc, self.SP.Nt_RF), dtype=np.complex128)
            self.W_RF = np.zeros((self.SP.Nr, self.SP.Nu, self.SP.Nc, self.SP.Nr_RF), dtype=np.complex128)



            self.AoD_pred = np.random.uniform(low=-1, high=1, size=(self.SP.Nc, self.SP.Nt_RF))
            self.AoA_pred = np.random.uniform(low=-1, high=1, size=(self.SP.Nu, self.SP.Nc, self.SP.Nr_RF))
            self.AoD_pred[0,0] = np.float64(0)
            self.SteeringMatrix_TX() # self.AoD_pred -> F_RF (ULA)
            self.SteeringMatrix_RX() # self.AoA_pred -> W_RF (ULA)


            self.num_state = 2 * self.SP.Nc * self.SP.Nu #
            ## num_action is defined for F_BB, W_BB, AoD_pred, AoA_pred.
            ## note that analog precoder is parametrized by angles
            self.num_action =( 2*(self.SP.Nt_RF * self.SP.Ns * self.SP.Nu *self.SP.Nc)
                            + 2*(self.SP.Nr_RF * self.SP.Ns_peruser * self.SP.Nu * self.SP.Nc)
                            + self.SP.Nc * self.SP.Nt_RF
                            + self.SP.Nu *  self.SP.Nc * self.SP.Nr_RF )

            self.num_action = self.SP.Nc * self.SP.Nr_RF

            # The position where the angle prediction begins in action array
            self.begin_angle = 2*(self.SP.Nt_RF * self.SP.Ns * self.SP.Nu *self.SP.Nc \
                            + self.SP.Nr_RF * self.SP.Ns_peruser * self.SP.Nu * self.SP.Nc)

            self.begin_angle = 0

    def reset(self):
        self.compute_SINR()


        return np.array([self.SINR, self.SP.D / self.SP.r_cell])

    def step(self, action):


        self.AoA_pred = action[:].data.numpy()
        self.AoA_pred = self.AoA_pred.reshape(self.SP.Nu, self.SP.Nc, self.SP.Nr_RF)

        self.AoD_pred[0,0] = np.float64(0)

        self.SteeringMatrix_TX()
        self.SteeringMatrix_RX()
        self.compute_SINR()

        h = self.SP.H[:, :, 0]
        WRF = self.W_RF[:, 0, 0, :]  # doesn't depend on TX

        # print(WRF.conj().T @ h)

        if self.SINR == 1:
            done = True
        else:
            done = False


        return np.array([self.SINR, self.SP.D / self.SP.r_cell]), self.SINR, done # next_state, reward, done



    def compute_SINR(self):
        self.SINR = np.zeros((self.SP.Nu, self.SP.Nc), dtype=np.float)

        for u in range(self.SP.Nu):
            for c in range(self.SP.Nc):
                interference = 0
                start = self.SP.Nt * (c * self.SP.Nu) + (u * self.SP.Nt) + (0)
                end = self.SP.Nt * (c * self.SP.Nu) + (u * self.SP.Nt) + (self.SP.Nt - 1)

                WBB = self.W_BB[:, :, u, c]  # doesn't depend on TX
                WBB[0,0] = 1.
                WRF = self.W_RF[:, u, c, :]  # doesn't depend on TX



                for u_prime in range(self.SP.Nu):
                    for c_prime in range(self.SP.Nc):
                        h = self.SP.H[:, start:end + 1, c_prime]
                        FBB = self.F_BB[:,:,u_prime,c_prime]
                        FBB[0,0] = 1.
                        FRF = self.F_RF[:,c_prime,:]
                        TX = FRF @ FBB
                        #TX = TX / np.linalg.norm(FRF @ FBB, ord='fro') # @ means matrix multiplication
                        #TX = TX * self.SP.power_lin / (self.SP.Nu * self.SP.Ns_peruser)
                        RX = WRF @ WBB

                        h_eff = WRF.conj().T @ h @ TX
                        ##
                        #WBB = np.linalg.inv(h_eff @ h_eff.conj().T + WRF.conj().T@WRF) @ h_eff
                        ##


                        power = RX.conj().T @ h @ TX @ TX.conj().T @ h.conj().T @ RX

                        power = WRF.conj().T @ h  @ h.conj().T @ WRF

                        if u_prime == u and c_prime == c:
                            signal = power
                        else:
                            interference += power

                        noise = self.SP.noise * WBB.conj().T @ WRF.conj().T @ WRF @ WBB

                self.SINR[u,c] = abs(signal / (interference+noise))

        self.SINR_sum = np.product(self.SINR)


    def SteeringMatrix_TX(self):
        for c in range(self.SP.Nc):
            for rf in range(self.SP.Nt_RF):
                self.F_RF[:, c:c+1, rf] = SteeringGen(self.AoD_pred[c, rf], self.SP.Nt)

    def SteeringMatrix_RX(self):
        for u in range(self.SP.Nu):
            for c in range(self.SP.Nc):
                for rf in range(self.SP.Nr_RF):
                    self.W_RF[:, u:u + 1, c, rf] = SteeringGen(self.AoA_pred[u, c, rf], self.SP.Nr)


