# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 21:00:45 2017

@author: dschaffner
"""
import numpy as np
import matplotlib.pylab as plt
from loadnpzfile import loadnpzfile
from calc_PE_SC import PE, CH, PE_dist, PE_calc_only
from collections import Counter
from math import factorial
import os

#calc_PESC_solarwind_chen.py
datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Galatic Dynamics Data\\DoublePendulum\\'
fileheader = 'PE_SC_DPDoubPen_LsMsEq1_grav1_ICC1_embeddelay5_999_delays'
npz='.npz'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_L1=datafile['PEsx1']
SCsx1_L1=datafile['SCsx1']
delayindex=datafile['delays']

maxmin_range_index1 = 50
maxmin_range_index2 = 700

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC0_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang0=datafile['PEsx1']
SCsx1_icang0=datafile['SCsx1']
SCmax_icang0 = np.max(SCsx1_icang0[maxmin_range_index1:maxmin_range_index2])
SCmin_icang0 = np.min(SCsx1_icang0[maxmin_range_index1:maxmin_range_index2])
SCrange_icang0 = SCmax_icang0-SCmin_icang0

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC1_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang1=datafile['PEsx1']
SCsx1_icang1=datafile['SCsx1']
SCmax_icang1 = np.max(SCsx1_icang1[maxmin_range_index1:maxmin_range_index2])
SCmin_icang1 = np.min(SCsx1_icang1[maxmin_range_index1:maxmin_range_index2])
SCrange_icang1 = SCmax_icang1-SCmin_icang1

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC2_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang2=datafile['PEsx1']
SCsx1_icang2=datafile['SCsx1']
SCmax_icang2 = np.max(SCsx1_icang2[maxmin_range_index1:maxmin_range_index2])
SCmin_icang2 = np.min(SCsx1_icang2[maxmin_range_index1:maxmin_range_index2])
SCrange_icang2 = SCmax_icang2-SCmin_icang2

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC3_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang3=datafile['PEsx1']
SCsx1_icang3=datafile['SCsx1']
SCmax_icang3 = np.max(SCsx1_icang3[maxmin_range_index1:maxmin_range_index2])
SCmin_icang3 = np.min(SCsx1_icang3[maxmin_range_index1:maxmin_range_index2])
SCrange_icang3 = SCmax_icang3-SCmin_icang3

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC4_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang4=datafile['PEsx1']
SCsx1_icang4=datafile['SCsx1']
SCmax_icang4 = np.max(SCsx1_icang4[maxmin_range_index1:maxmin_range_index2])
SCmin_icang4 = np.min(SCsx1_icang4[maxmin_range_index1:maxmin_range_index2])
SCrange_icang4 = SCmax_icang4-SCmin_icang4

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC5_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang5=datafile['PEsx1']
SCsx1_icang5=datafile['SCsx1']
SCmax_icang5 = np.max(SCsx1_icang5[maxmin_range_index1:maxmin_range_index2])
SCmin_icang5 = np.min(SCsx1_icang5[maxmin_range_index1:maxmin_range_index2])
SCrange_icang5 = SCmax_icang5-SCmin_icang5

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC6_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang6=datafile['PEsx1']
SCsx1_icang6=datafile['SCsx1']
SCmax_icang6 = np.max(SCsx1_icang6[maxmin_range_index1:maxmin_range_index2])
SCmin_icang6 = np.min(SCsx1_icang6[maxmin_range_index1:maxmin_range_index2])
SCrange_icang6 = SCmax_icang6-SCmin_icang6

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC7_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang7=datafile['PEsx1']
SCsx1_icang7=datafile['SCsx1']
SCmax_icang7 = np.max(SCsx1_icang7[maxmin_range_index1:maxmin_range_index2])
SCmin_icang7 = np.min(SCsx1_icang7[maxmin_range_index1:maxmin_range_index2])
SCrange_icang7 = SCmax_icang7-SCmin_icang7

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC8_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang8=datafile['PEsx1']
SCsx1_icang8=datafile['SCsx1']
SCmax_icang8 = np.max(SCsx1_icang8[maxmin_range_index1:maxmin_range_index2])
SCmin_icang8 = np.min(SCsx1_icang8[maxmin_range_index1:maxmin_range_index2])
SCrange_icang8 = SCmax_icang8-SCmin_icang8

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC9_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang9=datafile['PEsx1']
SCsx1_icang9=datafile['SCsx1']
SCmax_icang9 = np.max(SCsx1_icang9[maxmin_range_index1:maxmin_range_index2])
SCmin_icang9 = np.min(SCsx1_icang9[maxmin_range_index1:maxmin_range_index2])
SCrange_icang9 = SCmax_icang9-SCmin_icang9

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC10_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang10=datafile['PEsx1']
SCsx1_icang10=datafile['SCsx1']
SCmax_icang10 = np.max(SCsx1_icang10[maxmin_range_index1:maxmin_range_index2])
SCmin_icang10 = np.min(SCsx1_icang10[maxmin_range_index1:maxmin_range_index2])
SCrange_icang10 = SCmax_icang10-SCmin_icang10

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC11_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang11=datafile['PEsx1']
SCsx1_icang11=datafile['SCsx1']
SCmax_icang11 = np.max(SCsx1_icang11[maxmin_range_index1:maxmin_range_index2])
SCmin_icang11 = np.min(SCsx1_icang11[maxmin_range_index1:maxmin_range_index2])
SCrange_icang11 = SCmax_icang11-SCmin_icang11

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC12_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang12=datafile['PEsx1']
SCsx1_icang12=datafile['SCsx1']
SCmax_icang12 = np.max(SCsx1_icang12[maxmin_range_index1:maxmin_range_index2])
SCmin_icang12 = np.min(SCsx1_icang12[maxmin_range_index1:maxmin_range_index2])
SCrange_icang12 = SCmax_icang12-SCmin_icang12

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC13_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang13=datafile['PEsx1']
SCsx1_icang13=datafile['SCsx1']
SCmax_icang13 = np.max(SCsx1_icang13[maxmin_range_index1:maxmin_range_index2])
SCmin_icang13 = np.min(SCsx1_icang13[maxmin_range_index1:maxmin_range_index2])
SCrange_icang13 = SCmax_icang13-SCmin_icang13

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC14_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang14=datafile['PEsx1']
SCsx1_icang14=datafile['SCsx1']
SCmax_icang14 = np.max(SCsx1_icang14[maxmin_range_index1:maxmin_range_index2])
SCmin_icang14 = np.min(SCsx1_icang14[maxmin_range_index1:maxmin_range_index2])
SCrange_icang14 = SCmax_icang14-SCmin_icang14

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC15_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang15=datafile['PEsx1']
SCsx1_icang15=datafile['SCsx1']
SCmax_icang15 = np.max(SCsx1_icang15[maxmin_range_index1:maxmin_range_index2])
SCmin_icang15 = np.min(SCsx1_icang15[maxmin_range_index1:maxmin_range_index2])
SCrange_icang15 = SCmax_icang15-SCmin_icang15

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC16_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang16=datafile['PEsx1']
SCsx1_icang16=datafile['SCsx1']
SCmax_icang16 = np.max(SCsx1_icang16[maxmin_range_index1:maxmin_range_index2])
SCmin_icang16 = np.min(SCsx1_icang16[maxmin_range_index1:maxmin_range_index2])
SCrange_icang16 = SCmax_icang16-SCmin_icang16

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC17_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang17=datafile['PEsx1']
SCsx1_icang17=datafile['SCsx1']
SCmax_icang17 = np.max(SCsx1_icang17[maxmin_range_index1:maxmin_range_index2])
SCmin_icang17 = np.min(SCsx1_icang17[maxmin_range_index1:maxmin_range_index2])
SCrange_icang17 = SCmax_icang17-SCmin_icang17

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC18_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang18=datafile['PEsx1']
SCsx1_icang18=datafile['SCsx1']
SCmax_icang18 = np.max(SCsx1_icang18[maxmin_range_index1:maxmin_range_index2])
SCmin_icang18 = np.min(SCsx1_icang18[maxmin_range_index1:maxmin_range_index2])
SCrange_icang18 = SCmax_icang18-SCmin_icang18

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC19_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang19=datafile['PEsx1']
SCsx1_icang19=datafile['SCsx1']
SCmax_icang19 = np.max(SCsx1_icang19[maxmin_range_index1:maxmin_range_index2])
SCmin_icang19 = np.min(SCsx1_icang19[maxmin_range_index1:maxmin_range_index2])
SCrange_icang19 = SCmax_icang19-SCmin_icang19

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC20_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang20=datafile['PEsx1']
SCsx1_icang20=datafile['SCsx1']
SCmax_icang20 = np.max(SCsx1_icang20[maxmin_range_index1:maxmin_range_index2])
SCmin_icang20 = np.min(SCsx1_icang20[maxmin_range_index1:maxmin_range_index2])
SCrange_icang20 = SCmax_icang20-SCmin_icang20

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC21_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang21=datafile['PEsx1']
SCsx1_icang21=datafile['SCsx1']
SCmax_icang21 = np.max(SCsx1_icang21[maxmin_range_index1:maxmin_range_index2])
SCmin_icang21 = np.min(SCsx1_icang21[maxmin_range_index1:maxmin_range_index2])
SCrange_icang21 = SCmax_icang21-SCmin_icang21

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC22_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang22=datafile['PEsx1']
SCsx1_icang22=datafile['SCsx1']
SCmax_icang22 = np.max(SCsx1_icang22[maxmin_range_index1:maxmin_range_index2])
SCmin_icang22 = np.min(SCsx1_icang22[maxmin_range_index1:maxmin_range_index2])
SCrange_icang22 = SCmax_icang22-SCmin_icang22

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC23_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang23=datafile['PEsx1']
SCsx1_icang23=datafile['SCsx1']
SCmax_icang23 = np.max(SCsx1_icang23[maxmin_range_index1:maxmin_range_index2])
SCmin_icang23 = np.min(SCsx1_icang23[maxmin_range_index1:maxmin_range_index2])
SCrange_icang23 = SCmax_icang23-SCmin_icang23

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC24_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang24=datafile['PEsx1']
SCsx1_icang24=datafile['SCsx1']
SCmax_icang24 = np.max(SCsx1_icang24[maxmin_range_index1:maxmin_range_index2])
SCmin_icang24 = np.min(SCsx1_icang24[maxmin_range_index1:maxmin_range_index2])
SCrange_icang24 = SCmax_icang24-SCmin_icang24

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC25_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang25=datafile['PEsx1']
SCsx1_icang25=datafile['SCsx1']
SCmax_icang25 = np.max(SCsx1_icang25[maxmin_range_index1:maxmin_range_index2])
SCmin_icang25 = np.min(SCsx1_icang25[maxmin_range_index1:maxmin_range_index2])
SCrange_icang25 = SCmax_icang25-SCmin_icang25

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC26_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang26=datafile['PEsx1']
SCsx1_icang26=datafile['SCsx1']
SCmax_icang26 = np.max(SCsx1_icang26[maxmin_range_index1:maxmin_range_index2])
SCmin_icang26 = np.min(SCsx1_icang26[maxmin_range_index1:maxmin_range_index2])
SCrange_icang26 = SCmax_icang26-SCmin_icang26

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC27_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang27=datafile['PEsx1']
SCsx1_icang27=datafile['SCsx1']
SCmax_icang27 = np.max(SCsx1_icang27[maxmin_range_index1:maxmin_range_index2])
SCmin_icang27 = np.min(SCsx1_icang27[maxmin_range_index1:maxmin_range_index2])
SCrange_icang27 = SCmax_icang27-SCmin_icang27

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC28_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang28=datafile['PEsx1']
SCsx1_icang28=datafile['SCsx1']
SCmax_icang28 = np.max(SCsx1_icang28[maxmin_range_index1:maxmin_range_index2])
SCmin_icang28 = np.min(SCsx1_icang28[maxmin_range_index1:maxmin_range_index2])
SCrange_icang28 = SCmax_icang28-SCmin_icang28

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC29_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang29=datafile['PEsx1']
SCsx1_icang29=datafile['SCsx1']
SCmax_icang29 = np.max(SCsx1_icang29[maxmin_range_index1:maxmin_range_index2])
SCmin_icang29 = np.min(SCsx1_icang29[maxmin_range_index1:maxmin_range_index2])
SCrange_icang29 = SCmax_icang29-SCmin_icang29

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC30_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang30=datafile['PEsx1']
SCsx1_icang30=datafile['SCsx1']
SCmax_icang30 = np.max(SCsx1_icang30[maxmin_range_index1:maxmin_range_index2])
SCmin_icang30 = np.min(SCsx1_icang30[maxmin_range_index1:maxmin_range_index2])
SCrange_icang30 = SCmax_icang30-SCmin_icang30

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC31_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang31=datafile['PEsx1']
SCsx1_icang31=datafile['SCsx1']
SCmax_icang31 = np.max(SCsx1_icang31[maxmin_range_index1:maxmin_range_index2])
SCmin_icang31 = np.min(SCsx1_icang31[maxmin_range_index1:maxmin_range_index2])
SCrange_icang31 = SCmax_icang31-SCmin_icang31

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC32_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang32=datafile['PEsx1']
SCsx1_icang32=datafile['SCsx1']
SCmax_icang32 = np.max(SCsx1_icang32[maxmin_range_index1:maxmin_range_index2])
SCmin_icang32 = np.min(SCsx1_icang32[maxmin_range_index1:maxmin_range_index2])
SCrange_icang32 = SCmax_icang32-SCmin_icang32

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC33_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang33=datafile['PEsx1']
SCsx1_icang33=datafile['SCsx1']
SCmax_icang33 = np.max(SCsx1_icang33[maxmin_range_index1:maxmin_range_index2])
SCmin_icang33 = np.min(SCsx1_icang33[maxmin_range_index1:maxmin_range_index2])
SCrange_icang33 = SCmax_icang33-SCmin_icang33

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC34_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang34=datafile['PEsx1']
SCsx1_icang34=datafile['SCsx1']
SCmax_icang34 = np.max(SCsx1_icang34[maxmin_range_index1:maxmin_range_index2])
SCmin_icang34 = np.min(SCsx1_icang34[maxmin_range_index1:maxmin_range_index2])
SCrange_icang34 = SCmax_icang34-SCmin_icang34

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC35_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang35=datafile['PEsx1']
SCsx1_icang35=datafile['SCsx1']
SCmax_icang35 = np.max(SCsx1_icang35[maxmin_range_index1:maxmin_range_index2])
SCmin_icang35 = np.min(SCsx1_icang35[maxmin_range_index1:maxmin_range_index2])
SCrange_icang35 = SCmax_icang35-SCmin_icang35

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC36_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang36=datafile['PEsx1']
SCsx1_icang36=datafile['SCsx1']
SCmax_icang36 = np.max(SCsx1_icang36[maxmin_range_index1:maxmin_range_index2])
SCmin_icang36 = np.min(SCsx1_icang36[maxmin_range_index1:maxmin_range_index2])
SCrange_icang36 = SCmax_icang36-SCmin_icang36

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC37_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang37=datafile['PEsx1']
SCsx1_icang37=datafile['SCsx1']
SCmax_icang37 = np.max(SCsx1_icang37[maxmin_range_index1:maxmin_range_index2])
SCmin_icang37 = np.min(SCsx1_icang37[maxmin_range_index1:maxmin_range_index2])
SCrange_icang37 = SCmax_icang37-SCmin_icang37

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC38_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang38=datafile['PEsx1']
SCsx1_icang38=datafile['SCsx1']
SCmax_icang38 = np.max(SCsx1_icang38[maxmin_range_index1:maxmin_range_index2])
SCmin_icang38 = np.min(SCsx1_icang38[maxmin_range_index1:maxmin_range_index2])
SCrange_icang38 = SCmax_icang38-SCmin_icang38

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC39_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang39=datafile['PEsx1']
SCsx1_icang39=datafile['SCsx1']
SCmax_icang39 = np.max(SCsx1_icang39[maxmin_range_index1:maxmin_range_index2])
SCmin_icang39 = np.min(SCsx1_icang39[maxmin_range_index1:maxmin_range_index2])
SCrange_icang39 = SCmax_icang39-SCmin_icang39

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC40_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang40=datafile['PEsx1']
SCsx1_icang40=datafile['SCsx1']
SCmax_icang40 = np.max(SCsx1_icang40[maxmin_range_index1:maxmin_range_index2])
SCmin_icang40 = np.min(SCsx1_icang40[maxmin_range_index1:maxmin_range_index2])
SCrange_icang40 = SCmax_icang40-SCmin_icang40

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC41_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang41=datafile['PEsx1']
SCsx1_icang41=datafile['SCsx1']
SCmax_icang41 = np.max(SCsx1_icang41[maxmin_range_index1:maxmin_range_index2])
SCmin_icang41 = np.min(SCsx1_icang41[maxmin_range_index1:maxmin_range_index2])
SCrange_icang41 = SCmax_icang41-SCmin_icang41

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC42_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang42=datafile['PEsx1']
SCsx1_icang42=datafile['SCsx1']
SCmax_icang42 = np.max(SCsx1_icang42[maxmin_range_index1:maxmin_range_index2])
SCmin_icang42 = np.min(SCsx1_icang42[maxmin_range_index1:maxmin_range_index2])
SCrange_icang42 = SCmax_icang42-SCmin_icang42

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC43_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang43=datafile['PEsx1']
SCsx1_icang43=datafile['SCsx1']
SCmax_icang43 = np.max(SCsx1_icang43[maxmin_range_index1:maxmin_range_index2])
SCmin_icang43 = np.min(SCsx1_icang43[maxmin_range_index1:maxmin_range_index2])
SCrange_icang43 = SCmax_icang43-SCmin_icang43

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC44_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang44=datafile['PEsx1']
SCsx1_icang44=datafile['SCsx1']
SCmax_icang44 = np.max(SCsx1_icang44[maxmin_range_index1:maxmin_range_index2])
SCmin_icang44 = np.min(SCsx1_icang44[maxmin_range_index1:maxmin_range_index2])
SCrange_icang44 = SCmax_icang44-SCmin_icang44

SCmins=np.array([SCmin_icang0,
                 SCmin_icang1,
                 SCmin_icang2,
                 SCmin_icang3,
                 SCmin_icang4,
                 SCmin_icang5,
                 SCmin_icang6,
                 SCmin_icang7,
                 SCmin_icang8,
                 SCmin_icang9,
                 SCmin_icang10,
                 SCmin_icang11,
                 SCmin_icang12,
                 SCmin_icang13,
                 SCmin_icang14,
                 SCmin_icang15,
                 SCmin_icang16,
                 SCmin_icang17,
                 SCmin_icang18,
                 SCmin_icang19,
                 SCmin_icang20,
                 SCmin_icang21,
                 SCmin_icang22,
                 SCmin_icang23,
                 SCmin_icang24,
                 SCmin_icang25,
                 SCmin_icang26,
                 SCmin_icang27,
                 SCmin_icang28,
                 SCmin_icang29,
                 SCmin_icang30,
                 SCmin_icang31,
                 SCmin_icang32,
                 SCmin_icang33,
                 SCmin_icang40,
                 SCmin_icang41,
                 SCmin_icang42,
                 SCmin_icang43,
                 SCmin_icang44])

SCmaxes=np.array([SCmax_icang0,
                 SCmax_icang1,
                 SCmax_icang2,
                 SCmax_icang3,
                 SCmax_icang4,
                 SCmax_icang5,
                 SCmax_icang6,
                 SCmax_icang7,
                 SCmax_icang8,
                 SCmax_icang9,
                 SCmax_icang10,
                 SCmax_icang11,
                 SCmax_icang12,
                 SCmax_icang13,
                 SCmax_icang14,
                 SCmax_icang15,
                 SCmax_icang16,
                 SCmax_icang17,
                 SCmax_icang18,
                 SCmax_icang19,
                 SCmax_icang20,
                 SCmax_icang21,
                 SCmax_icang22,
                 SCmax_icang23,
                 SCmax_icang24,
                 SCmax_icang25,
                 SCmax_icang26,
                 SCmax_icang27,
                 SCmax_icang28,
                 SCmax_icang29,
                 SCmax_icang30,
                 SCmax_icang31,
                 SCmax_icang32,
                 SCmax_icang33,
                 SCmax_icang40,
                 SCmax_icang41,
                 SCmax_icang42,
                 SCmax_icang43,
                 SCmax_icang44])

SCranges=np.array([SCrange_icang0,
                 SCrange_icang1,
                 SCrange_icang2,
                 SCrange_icang3,
                 SCrange_icang4,
                 SCrange_icang5,
                 SCrange_icang6,
                 SCrange_icang7,
                 SCrange_icang8,
                 SCrange_icang9,
                 SCrange_icang10,
                 SCrange_icang11,
                 SCrange_icang12,
                 SCrange_icang13,
                 SCrange_icang14,
                 SCrange_icang15,
                 SCrange_icang16,
                 SCrange_icang17,
                 SCrange_icang18,
                 SCrange_icang19,
                 SCrange_icang20,
                 SCrange_icang21,
                 SCrange_icang22,
                 SCrange_icang23,
                 SCrange_icang24,
                 SCrange_icang25,
                 SCrange_icang26,
                 SCrange_icang27,
                 SCrange_icang28,
                 SCrange_icang29,
                 SCrange_icang30,
                 SCrange_icang31,
                 SCrange_icang32,
                 SCrange_icang33,
                 SCrange_icang40,
                 SCrange_icang41,
                 SCrange_icang42,
                 SCrange_icang43,
                 SCrange_icang44])

lyapanovs = np.array([0.000024,
                      0.000004,
                      0.000263,
                      0.000154,
                      0.00046,
                      0.000398,
                      0.0025,
                      0.000486,
                      0.000418,
                      0.000272,
                      0.000607,
                      0.000741,
                      0.000527,
                      0.000516,
                      0.001702,
                      0.0176,
                      0.0146,
                      0.01556,
                      0.02123,
                      0.05307,
                      0.0316,
                      0.07181,
                      0.0949,
                      0.04082,
                      0.1304,
                      0.21487,
                      0.1134,
                      0.2585,
                      0.5438,
                      0.3633,
                      0.21081,
                      0.4466,
                      0.4103,
                      0.6104,
                      0.00325,
                      0.00373,
                      0.00456,
                      0.00463,
                      0.0086])

        
plt.rc('axes',linewidth=2.0)
plt.rc('xtick.major',width=2.0)
plt.rc('ytick.major',width=2.0)
plt.rc('xtick.minor',width=2.0)
plt.rc('ytick.minor',width=2.0)
plt.rc('lines',markersize=8,markeredgewidth=0.0,linewidth=2.0)
#plt.rcParams['ps.fonttype'] = 42
fig=plt.figure(num=1,figsize=(9,7),dpi=600,facecolor='w',edgecolor='k')
left  = 0.16  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.08  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.0   # the amount of width reserved for blank space between subplots
hspace = 0.0   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
ax1=plt.subplot(3,2,1)
ax1.tick_params(axis='x',direction='inout',top=True)
ax1.tick_params(axis='y',direction='inout',top=True)

logmaxes = np.log10(SCmaxes)
logmins = np.log10(SCmins)
loglya = np.log(lyapanovs)
#linear regression
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(loglya[3:],SCmaxes[3:])
print 'C maxes intercept ',intercept
print 'C maxes slope ',slope

#plt.figure(1)
plt.plot(loglya,SCmaxes,'o')
plt.plot(loglya,(loglya*slope)+intercept,color='red',label=r'Linear Fit, $R^{2}=$'+str(np.round(r_value**2,3)))
plt.ylabel(r'$C_{max}$',fontsize=15)
plt.xlabel('Log Lyapanov',fontsize=15)

ax1=plt.subplot(3,2,2)
ax1.tick_params(axis='x',direction='inout',top=True)
ax1.tick_params(axis='y',direction='inout',top=True)
plt.plot(lyapanovs,SCmaxes,'o')
lya_axis = (np.arange(25434)*0.000024)+0.000024
plt.plot(lya_axis,intercept+(slope*np.log(lya_axis)))
plt.xlabel('Lyapanov',fontsize=15)
ax1.set_yticklabels([])


slope, intercept, r_value, p_value, std_err = stats.linregress(loglya[3:],SCmins[3:])
print 'C mins intercept ',intercept
print 'C mins slope ',slope

ax1=plt.subplot(3,2,3)
ax1.tick_params(axis='x',direction='inout',top=True)
ax1.tick_params(axis='y',direction='inout',top=True)
plt.plot(loglya,SCmins,'o')
plt.plot(loglya,(loglya*slope)+intercept,color='red',label=r'Linear Fit, $R^{2}=$'+str(np.round(r_value**2,3)))
plt.ylabel(r'$C_{min}$',fontsize=15)
plt.xlabel('Log Lyapanov',fontsize=15)

ax1=plt.subplot(3,2,4)
ax1.tick_params(axis='x',direction='inout',top=True)
ax1.tick_params(axis='y',direction='inout',top=True)
plt.plot(lyapanovs,SCmins,'o')
lya_axis = (np.arange(25434)*0.000024)+0.000024
plt.plot(lya_axis,intercept+(slope*np.log(lya_axis)))
plt.xlabel('Lyapanov',fontsize=15)
ax1.set_yticklabels([])

slope, intercept, r_value, p_value, std_err = stats.linregress(loglya[3:],SCranges[3:])
print 'C range intercept ',intercept
print 'C range slope ',slope

ax1=plt.subplot(3,2,5)
ax1.tick_params(axis='x',direction='inout',top=True)
ax1.tick_params(axis='y',direction='inout',top=True)
plt.plot(loglya,SCranges,'o')
plt.plot(loglya,(loglya*slope)+intercept,color='red',label=r'Linear Fit, $R^{2}=$'+str(np.round(r_value**2,3)))
plt.ylabel(r'$|C_{max}-C_{min}|$',fontsize=15)
plt.xlabel('Log Lyapanov',fontsize=15)


ax1=plt.subplot(3,2,6)
ax1.tick_params(axis='x',direction='inout',top=True)
ax1.tick_params(axis='y',direction='inout',top=True)
plt.plot(lyapanovs,SCranges,'o')
lya_axis = (np.arange(25434)*0.000024)+0.000024
plt.plot(lya_axis,intercept+(slope*np.log(lya_axis)))
plt.xlabel('Lyapanov',fontsize=15)
ax1.set_yticklabels([])

savefilename='Cmeasures_vs_Lyapunov_3.eps'
#savefilename='PE_galpy0718_1000timesteps_all_orbits.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')
plt.clf()
plt.close()

"""
tmax, dt = 100, 0.001
t = np.arange(0, tmax+dt, dt)
timeindex = delayindex*0.001

import matplotlib.cm as cm
colors = np.zeros([20,4])
for i in np.arange(20):
    c = cm.spectral(i/20.,1)
    colors[i,:]=c
points = ['o','v','s','p','*','h','^','D','+','>','H','d','x','<']
        
plt.rc('axes',linewidth=0.75)
plt.rc('xtick.major',width=0.75)
plt.rc('ytick.major',width=0.75)
plt.rc('xtick.minor',width=0.75)
plt.rc('ytick.minor',width=0.75)
plt.rc('lines',markersize=2,markeredgewidth=0.0)

plt.rc('lines',markersize=1.5,markeredgewidth=0.0)
fig=plt.figure(num=1,figsize=(3.5,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.2  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.97      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(1,1,1)
plt.semilogx(delayindex,SCsx1_Lp1,color='black',label='L=0.1')
plt.semilogx(delayindex,SCsx1_Lp5,color='blue',label='L=0.5')
plt.semilogx(delayindex,SCsx1_L1,color='green',label='L=1.0')
plt.semilogx(delayindex,SCsx1_L10,color='red',label='L=10.0')
plt.semilogx(delayindex,SCsx1_L100,color='orange',label='L=100.0')




#plt.xticks(np.array([1,40,80,120,160,200,240]),[1,40,80,120,160,200,240],fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel('Delay Steps',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Statistical Complexity',fontsize=9)
#plt.xlim(1,250)
plt.ylim(0,0.5)
plt.legend(loc='upper left',fontsize=5,frameon=False,handlelength=5)

savefilename='SC_DPx1_ICC1_Lscan_0p1to100.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')
"""