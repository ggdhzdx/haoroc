import os, linecache, shlex
import subprocess as sp
import numpy as np

def st_excited(file):          
    f_1 = open(file,'r',encoding='utf-8')
    line = f_1.readlines()
    
    S1_index = []
    T1_index = []
    for index,lines in enumerate(line):
        if "Singlet-" in lines:
            s1_s0 = index
            S1_index.append(s1_s0)
        if "Triplet-" in lines:
            t1_s0 = index
            T1_index.append(t1_s0)
            
    S1_ex_num = np.loadtxt(file,skiprows=S1_index[0],max_rows=1,usecols=(2),encoding='utf-8',dtype=str)
    T1_ex_num = np.loadtxt(file,skiprows=T1_index[0],max_rows=1,usecols=(2),encoding='utf-8',dtype=str)
    return [int(str(S1_ex_num)[0:-1]),int(str(T1_ex_num)[0:-1])]
	
def gen_cub(logfile,fchkfile):
    ex_num = st_excited(logfile)
    exe_file = 'Multiwfn '+ str(fchkfile)
    input_data_s1 = f"18\n1\n{logfile}\n{int(ex_num[0])}\n1\n3\n10\n1\n11\n1\n0\n0\n0\nq\n"
    input_data_t1 = f"18\n1\n{logfile}\n{int(ex_num[1])}\n1\n3\n10\n1\n11\n1\n0\n0\n0\nq\n"

    process = sp.Popen(shlex.split(exe_file),stdin=sp.PIPE,stdout=sp.PIPE,stderr=sp.PIPE)
    process.stdin.write(input_data_s1.encode())
    process.stdin.flush()
    data=process.stdout.read().decode()  
    os.rename('hole.cub', str(logfile)[0:-11]+'_s1_hole.cub')
    os.rename('electron.cub', str(logfile)[0:-11]+'_s1_elec.cub')

    process = sp.Popen(shlex.split(exe_file),stdin=sp.PIPE,stdout=sp.PIPE,stderr=sp.PIPE)
    process.stdin.write(input_data_t1.encode())
    process.stdin.flush()
    data=process.stdout.read().decode()
    os.rename('hole.cub', str(logfile)[0:-11]+'_t1_hole.cub')
    os.rename('electron.cub', str(logfile)[0:-11]+'_t1_elec.cub')


	
logFiles = []
for filename in os.listdir('.'): 
    if filename.endswith('.log'):
        logFiles.append(filename)
     
for i in logFiles:
    gen_cub(i,str(i)[0:-4]+'.fchk')	
