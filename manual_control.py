import sys

import numpy as np

import survival
import gymnasium as gym


# importing libraries
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
 

class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        # gym stuff
        self.env = survival.SurvivalEnv()
        # self.env = gym.make('Ant-v4', render_mode="human")
        self.env.render_mode = "human"

        self.n_actions = self.env.action_space.shape[-1]
        self.restart = True

        # qt stuff
        self.setWindowTitle("manual control")
        self.idx_to_name = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q']
        self.setGeometry(100, 100, 300, 60*self.n_actions+80)
        self.spins = []
        for idx in range(self.n_actions):
            # creating spin box
            spin = QDoubleSpinBox(self)
            spin.setGeometry(10, 60*idx, 90, 50)
            spin.setRange(-1, 1)
            spin.setSingleStep(0.05)
            spin.setPrefix("("+self.idx_to_name[idx]+") ")
            self.spins.append(spin)
        self.reset_button = QPushButton("reset", parent=self)
        self.reset_button.setGeometry(10, 60*self.n_actions, 100, 50)
        self.reset_button.clicked.connect(self.reset)
        self.show()

        # timer
        self.timer = QtCore.QTimer()
        self.timer.start(10)
        self.timer.timeout.connect(self.step)

    def reset(self):
        self.env.close()
        self.env = survival.SurvivalEnv()
        # self.env = gym.make('Ant-v4')

        self.n_actions = self.env.action_space.shape[-1]
        self.restart = True

    def step(self):
        if self.restart:
            self.obs = self.env.reset()
            # self.env.render_mode = "human"

        action = np.array([spin.value() for spin in self.spins])
        [spin.setValue(0) for spin in self.spins]
        print('step', action)
        self.obs, self.reward, self.restart, _, self.info = self.env.step(action)
        self.env.render()

# create pyqt5 app
App = QApplication(sys.argv)
 
# create the instance of our Window
window = Window()
 
# start the app
sys.exit(App.exec())