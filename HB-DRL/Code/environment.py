from channel import SteeringGen
from channel import mmWaveChannel
import numpy as np
import torch




class radio_env:
    def __init__(self):
        self.SP = mmWaveChannel()
        self.SP.generate()

        self.state = 0

        self.F_BB = np.random.uniform(size=(self.SP.Nt_RF, self.SP.Ns, self.SP.Nu, self.SP.Nc)) \
                 + 1j*np.random.random(size=(self.SP.Nt_RF, self.SP.Ns, self.SP.Nu, self.SP.Nc))  # a BS <-> a user Nr X Nt
        self.W_BB = np.random.uniform(size=(self.SP.Nr_RF, self.SP.Ns_peruser, self.SP.Nu, self.SP.Nc)) \
                 + 1j*np.random.uniform(size=(self.SP.Nr_RF, self.SP.Ns_peruser, self.SP.Nu, self.SP.Nc))

        
        self.F_RF = np.zeros((self.SP.Nt, self.SP.Nc, self.SP.Nt_RF), dtype=np.complex128)
        self.W_RF = np.zeros((self.SP.Nr, self.SP.Nu, self.SP.Nc, self.SP.Nr_RF), dtype=np.complex128)

        self.AoD_pred = np.random.uniform(low=-1, high=1, size=(self.SP.Nc, self.SP.Nt_RF))
        self.AoA_pred = np.random.uniform(low=-1, high=1, size=(self.SP.Nu, self.SP.Nc, self.SP.Nr_RF))
        self.SteeringMatrix_TX() # self.AoD_pred -> F_RF (ULA)
        self.SteeringMatrix_RX() # self.AoA_pred -> W_RF (ULA)

        self.num_state = self.SP.Nc * self.SP.Nu
        ## num_action is defined for F_BB, W_BB, AoD_pred, AoA_pred.
        ## note that analog precoder is parametrized by angles
        self.num_action =( 2*(self.SP.Nt_RF * self.SP.Ns * self.SP.Nu *self.SP.Nc) 
                        + 2*(self.SP.Nr_RF * self.SP.Ns_peruser * self.SP.Nu * self.SP.Nc) 
                        + self.SP.Nc * self.SP.Nt_RF 
                        + self.SP.Nu *  self.SP.Nc * self.SP.Nr_RF )

        # The position where the angle prediction begins in action array
        self.begin_angle = 2*(self.SP.Nt_RF * self.SP.Ns * self.SP.Nu *self.SP.Nc \
                        + self.SP.Nr_RF * self.SP.Ns_peruser * self.SP.Nu * self.SP.Nc)

    def reset(self):
        self.compute_SINR()

        return np.array(self.SINR)

    def step(self, action):
        FBB_idx = self.SP.Nt_RF * self.SP.Ns * self.SP.Nu * self.SP.Nc
        WBB_idx = self.SP.Nr_RF * self.SP.Ns_peruser * self.SP.Nu * self.SP.Nc
        AoD_idx = self.SP.Nc * self.SP.Nt_RF
        AoA_idx = self.SP.Nu *  self.SP.Nc * self.SP.Nr_RF

        # chop the long vector into 4 parts to make matrices

        self.F_BB = action[0:FBB_idx].data.numpy() + 1j*action[FBB_idx:2*FBB_idx].data.numpy()
        self.F_BB = self.F_BB.reshape(self.SP.Nt_RF, self.SP.Ns, self.SP.Nu, self.SP.Nc)

        self.W_BB = action[2*FBB_idx:2*FBB_idx+WBB_idx].data.numpy() + 1j*action[2*FBB_idx+WBB_idx:2*FBB_idx+2*WBB_idx].data.numpy()
        self.W_BB = self.W_BB.reshape(self.SP.Nr_RF, self.SP.Ns_peruser, self.SP.Nu, self.SP.Nc)

        self.AoD_pred = action[2*FBB_idx+2*WBB_idx:2*FBB_idx+2*WBB_idx+AoD_idx].data.numpy()
        self.AoD_pred = self.AoD_pred.reshape(self.SP.Nc, self.SP.Nt_RF)

        self.AoA_pred = action[2*FBB_idx+2*WBB_idx+AoD_idx:2*FBB_idx+2*WBB_idx+AoD_idx+AoA_idx].data.numpy()
        self.AoA_pred = self.AoA_pred.reshape(self.SP.Nu, self.SP.Nc, self.SP.Nr_RF)
        self.SteeringMatrix_TX()
        self.SteeringMatrix_RX()
        self.compute_SINR()


        done = False


        return 10*np.log10(self.SINR), 10*np.log10(self.SINR_sum), done # next_state, reward, done



    def compute_SINR(self):
        self.SINR = np.zeros((self.SP.Nu, self.SP.Nc), dtype=np.float)

        for u in range(self.SP.Nu):
            for c in range(self.SP.Nc):
                interference = 0
                start = self.SP.Nt * (c * self.SP.Nu) + (u * self.SP.Nt) + (0)
                end = self.SP.Nt * (c * self.SP.Nu) + (u * self.SP.Nt) + (self.SP.Nt - 1)

                WBB = self.W_BB[:, :, u, c]  # doesn't depend on TX
                WRF = self.W_RF[:, u, c, :]  # doesn't depend on TX



                for u_prime in range(self.SP.Nu):
                    for c_prime in range(self.SP.Nc):
                        h = self.SP.H[:, start:end + 1, c_prime]
                        FBB = self.F_BB[:,:,u_prime,c_prime]
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

                        power = WBB.conj().T @ WRF.conj().T @ h @ FRF @ FBB @ FBB.conj().T @ FRF.conj().T @ h.conj().T @ WRF @ WBB

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
                self.F_RF[:, c:c + 1, rf] = SteeringGen(self.AoD_pred[c, rf], self.SP.Nt)

    def SteeringMatrix_RX(self):
        for u in range(self.SP.Nu):
            for c in range(self.SP.Nc):
                for rf in range(self.SP.Nr_RF):
                    self.W_RF[:, u:u + 1, c, rf] = SteeringGen(self.AoA_pred[u, c, rf], self.SP.Nr)


