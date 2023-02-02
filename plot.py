import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import resultmanager


def oct25th2022():
    #filename= 'MCI_FDG-AV45k20.with.APOE'
    filename = 'PRS.with.APOE'
    ra = resultmanager.ResultAnalyzer(filename)
    ra.average_performance()


def correlation():
    pass

oct25th2022()
