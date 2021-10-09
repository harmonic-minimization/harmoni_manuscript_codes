"""
-----------------------------------------------------------------------
Harmoni: a Novel Method for Eliminating Spurious Neuronal Interactions due to the Harmonic Components in Neuronal Data
Mina Jamshidi Idaji, Juanli Zhang, Tilman Stephani, Guido Nolte, Klaus-Robert Mueller, Arno Villringer, Vadim V. Nikulin
https://doi.org/10.1101/2021.10.06.463319
-----------------------------------------------------------------------
(c) Mina Jamshidi (minajamshidi91@gmail.com) @ Neurolgy Dept, MPI CBS, 2021
https://github.com/minajamshidi
(c) please cite the above paper in case of using this code for your research

License: MIT License
-----------------------------------------------------------------------
"""

import os
import pandas
import numpy as np


def select_subjects(age_gr, gender_gr, handedness_gr, meta_file_path):
    # meta_file_path = '/data/pt_02076/LEMON/INFO/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv'
    # meta = pandas.read_csv(meta_file_path, sep=',')
    age = id_meta(meta_file_path, value='Age')
    id = id_meta(meta_file_path, value='ID')
    gender = id_meta(meta_file_path, value='Gender')
    handedness = id_meta(meta_file_path, value='Handedness')
    if age_gr == 'young':
        ind_age = (age == '20-25') + (age == '25-30') + (age == '30-35')
        ind_age = ind_age > 0
    if gender_gr == 'male':
        ind_gender = gender == 2
    ind_handedness = handedness == handedness_gr

    ind_final = (ind_handedness+0) + (ind_age+0) + (ind_gender+0)
    ind_final = ind_final == 3

    return id[ind_final]


def id_meta(meta_file_path, value=None):
    # meta_file_path = '/data/pt_02076/LEMON/INFO/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv'
    meta = pandas.read_csv(meta_file_path, sep=',')
    if value is None:
        return meta
    else:
        value_col = [i for i in range(10) if value in meta.columns[i]]
        if len(value_col):
            return meta.values[:, value_col]
        else:
            print('value is not valid!')