"""
The purpose of these commands was to fix an infinity symbol issue. 
I do not plan on making this a generic program, but something to reference to.
Created by Carlos A. Cartagena-Sanchez
"""
##################################################################################
""" You will have to adjust the max_Run_Number in line 23. 
    Also, you will have to change the filename variable on line 25-26."""
##################################################################################
import fileinput

def infinity_Change(filename, max_Value):

    with fileinput.FileInput(filename, inplace = True, backup = '.bak') as fi:
        for line in fi:
            print(line.replace(u"\u221E",str(max_Value)),end='')
    
    return None

def Loop_Infinity(file_Path, pico_Number):
    
    max_Value = 1 #This is true for the magnetics, not the monitor values.
    max_Run_Number = 17 #for my current situation 03/29/2019
    for run in range(1,max_Run_Number + 1):
        filename = (file_Path + "pico" + 
                    str(pico_Number) + "/20190423-0001 (" + str(run) + ").txt")
        infinity_Change(filename, max_Value)

    return None

file_Path = 'C:\\Users\\dschaffner\\Dropbox\\From OneDrive\\BM2X\\Data Storage\\04232019\\'
##################################################################################
""" The line below can be loop for the four picoscopes."""
##################################################################################

Loop_Infinity(file_Path, pico_Number = 4)