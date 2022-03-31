# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 13:31:06 2022

@author: u0103342
"""

import pandas as pd

volgorde=pd.read_excel(r'volgorde.xlsx', header=None, names=['Volgorde'])
dimensies=pd.read_csv(r'bakjes.csv', delimiter=";", usecols=lambda c: c not in ['Element', 'Property', 'Tol -', 'Tol +'])
dimensies=dimensies.T
dimensies.columns=['Plane1','Plane2', 'Plane3', 'Plane4', 'Plane4', 'Distance']


