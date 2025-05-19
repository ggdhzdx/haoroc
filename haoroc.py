#!/usr/bin/env python

import argparse
import numpy as np
import os
import sys
import shutil
import subprocess
import threading
import time
from threading import Thread
import pandas as pd
import re
from collections import defaultdict
import matplotlib.pyplot as plt 


# step 1 prepare input file, check out file, check molden file
# if molden file is not exist, then generate molden file
# step 2 indentify the heavy atom
# step 3 calculate the hole and electron of each heavy atom
# step 4 read the cube file and calculate the angle density of each heavy atom
# step 5 extract soc value from out file
parser=argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                               description='compute hole and electron density and the rotation on heavy atom\n'
                                           'require orca out and gbw/molden file with tddft calculation\n')
parser.add_argument("-H",dest='heavy',type=str,default='',
                         help='set the heavy atom threshold, e.g. 18\n'
                              'If not specified, atoms from the second period and above will be treated as heavy atoms.\n')
parser.add_argument("-g",dest='grid',type=str,default='4;0.1',
                         help='set the grid of cube file e.g. 3 for fine grid\n'
                              'default: 4;0.1 for 0.1 angstrom grid with 4 points\n')
parser.add_argument("--st",dest='st',type=str,default='1,1',
                         help='set the state no of singlets and triplets\n')
parser.add_argument('inputfile',nargs='+',help='the input structures should be gaussian log, or orca out')
parser.add_argument("--version",action="version",version='%(prog)s 1.0')
args=parser.parse_args()


class HeavyAtomOrbRotCompos:
    def __init__(self, inputfile, heavy_atom_th,singlet_triplet="1,1"):
        self.inputfile = inputfile
        self.load_parameters()
        self.cube_template=None
        self.siglet_sn=singlet_triplet.split(',')[0]
        self.triplet_sn=singlet_triplet.split(',')[1]
        self.basename = os.path.splitext(inputfile)[0]
        os.makedirs(self.basename,exist_ok=True)
        self.prog_type = ''
        self.prepare_input_file()
        self.read_coords()
        self.check_heavy_atom(heavy_atom_th=int(heavy_atom_th))
        self.calculate_he_cub()
        self.calc_cube()
        # self.extract_heavy_atom()
        # self.extract_soc()
        # self.extract_cube()
        # self.calculate_angle_density()
        # self.calculate_rotation()
    
    def load_parameters(self):
        """
        Load the parameters .
        """ 
        self.an2elem = {
            1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
            11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca',
            21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
            31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr',
            41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
            51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd',
            61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb',
            71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
            81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th',
            91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm',
            101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db', 106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt',
            110: 'Ds', 111: 'Rg', 112: 'Cn', 113: 'Nh', 114: 'Fl', 115: 'Mc', 116: 'Lv', 117: 'Ts', 118: 'Og'
        }
        self.elem2an = {v: k for k, v in self.an2elem.items()}  
        self.elem2n = {v: self.__z2n(v) for k, v in self.elem2an.items()}
        self.elem2esoc = {"C":28, "N":52, "O":151, "Cl":586, "K":38, "Cr":223, "Cu":828, "Br":2460, "Mo":552,
                          "Ru":990, "Rh":1212, "I":5060, "W":2089, "Re":2200, "Os":2500, "Pt":4000, "Au":5091,
                          "Tl": 7790, "Pb":7800, "U":2000}
        
    def __z2n(self,atomic_number):
        if atomic_number <= 2:
            return 1
        elif atomic_number <= 10:
            return 2
        elif atomic_number <= 18:
            return 3
        elif atomic_number <= 36:
            return 4
        elif atomic_number <= 54:
            return 5
        elif atomic_number <= 86:
            return 6
        elif atomic_number <= 118:
            return 7
        else:
            return None  # 对于超过118的原子序数，当前的周期表没有定义

    def __read_atomic_numbers(self):
        atomic_numbers = []
        wfnfile =  open(self.wfn_file, 'r')
        read_numbers = False
        if self.prog_type == 'gau':
            total_numbers = 0
            for line in wfnfile:
                if 'Atomic numbers' in line:
                    read_numbers = True
                    parts = line.split('N=')
                    if len(parts) > 1:
                        total_numbers = int(parts[1])
                    continue
                if read_numbers:
                    atomic_numbers += [int(i) for i in line.strip().split()]
                    if len(atomic_numbers) == total_numbers:
                        break
        elif self.prog_type == 'orca':
            for line in wfnfile:
                if '[Atoms]' in line:
                    read_numbers = True  # We're in the Atoms section
                    continue  # Skip the current line with '[Atoms]'
                # Check if we have reached the end of the Atoms section
                if read_numbers and ('[' in line and ']' in line):
                    break  # We've reached another section, so stop processing
                # If we're in the Atoms section, process the line
                if read_numbers:
                    parts = line.strip().split()  # Split the line into parts
                    if len(parts) > 2:  # Check if there are enough parts to read the third number
                        atomic_numbers.append(int(parts[2]))
        wfnfile.close()
        return atomic_numbers

    def prepare_input_file(self):
        """
        准备输入文件,检查文件类型,如果是高斯log文件则检查fchk文件是否存在，如果是orca out文件则检查molden文件。
        
        :param inputfile: 输入文件名kk
        """
        # 遍历文件前几行，检查文件类型
        inp = open(self.inputfile, 'r')
        i = 0
        for line in inp:
            if i > 10:
                break
            if 'Entering Gaussian System' in line:
                self.prog_type = 'gau'
            if '* O   R   C   A *' in line:
                self.prog_type = 'orca'
        if self.prog_type == 'gau':
            fchkfile = self.basename + '.fchk'
            if not os.path.exists(fchkfile):
                chkfile = self.basename + '.chk'
                if not os.path.exists(chkfile):
                    print(f'Error: {chkfile} not found')
                    sys.exit()
                else:
                    subprocess.run(['formchk', chkfile])
            if os.path.exists(fchkfile):
                self.wfn_file = fchkfile
            else:
                print(f'Error: {fchkfile} not found')
                sys.exit()
        elif self.prog_type == 'orca':
            moldenfile = self.basename + '.molden'
            if not os.path.exists(moldenfile):
                gbwfile = self.basename + '.gbw'
                if not os.path.exists(gbwfile):
                    print(f'Error: {gbwfile} not found')
                    sys.exit()
                else:
                    subprocess.run(['orca_2mkl', self.basename, '-molden'])
                    shutil.move(self.basename + '.molden.input', moldenfile)
            if os.path.exists(moldenfile):
                self.wfn_file = moldenfile
            else:
                print(f'Error: {moldenfile} not found')
                sys.exit()
        else:
            print(f'Error: {self.inputfile} is not a Gaussian log or Orca out file')
            sys.exit()

    def read_coords(self):
        """
        读取原子坐标.
        """
        self.coords = []
        wfnfile = open(self.wfn_file, 'r')
        read_coords = False
        if self.prog_type == 'gau':
            for line in wfnfile:
                if 'Current cartesian coordinates' in line:
                    read_coords = True
                    continue
                if read_coords:
                    parts = line.strip().split()
                    if len(parts) == 4:
                        self.coords.append(parts[1:])
                    else:
                        break
        elif self.prog_type == 'orca':
            for line in wfnfile:
                if '[Atoms]' in line:
                    read_coords = True
                    continue
                if read_coords:
                    parts = line.strip().split()
                    if len(parts) > 2:
                        self.coords.append(parts[3:])
                    else:
                        break

    def check_heavy_atom(self, heavy_atom_index=None, heavy_atom_th=18):
        """
        检查重原子的索引.
        """
        if heavy_atom_index is not None:
            return [i for i in heavy_atom_index.split(',')]
        else:
            self.atomic_numbers = self.__read_atomic_numbers()
        heavy_atom_index = [str(index+1) for index, value in enumerate(self.atomic_numbers) if value >= heavy_atom_th]
        print(f'Heavy atoms found: {",".join(heavy_atom_index)}')
        heavy_atom_label = ",".join([self.an2elem[self.atomic_numbers[int(index)-1]]+":"+index for index in heavy_atom_index])
        self.heavy_atom_index = heavy_atom_index 
        self.heavy_atom_symbol = [self.an2elem[self.atomic_numbers[int(index)-1]] for index in heavy_atom_index]
        self.heavy_atom_sn = [self.an2elem[self.atomic_numbers[int(index)-1]]+index for index in heavy_atom_index]
        self.heavy_atom_z = [self.atomic_numbers[int(index)-1] for index in heavy_atom_index]
        self.heavy_atom_n = [self.__z2n(self.atomic_numbers[int(index)-1]) for index in heavy_atom_index]
        print(f'Heavy atoms found: {heavy_atom_label}')
        df = pd.DataFrame({'index':self.heavy_atom_index,'symbol':self.heavy_atom_symbol,'sn':self.heavy_atom_sn,'z':self.heavy_atom_z,'n':self.heavy_atom_n},dtype=object)
        df['z4n3'] = (df['z'] ** 4) / (df['n'] ** 3)
        df['esoc'] = df['symbol'].map(self.elem2esoc)
        self.df = df.set_index('index')

    def __mtimer(self,task_name):
        t = 0
        thread = threading.current_thread()
        while getattr(thread, "do_run", True):
            print(f"\rRun Multiwfn for {task_name} ... {t}s", end=" ")
            sys.stdout.flush()
            time.sleep(0.5)
            t += 0.5

    def __gen_multiwfn_input(self,index,multi,nstate):
        """
        生成Multiwfn输入文件.
        """
        if index == '0':
            index = '1-10000'
        with open(self.basename + '.minp', 'w') as mwfile:
            mwfile.write('6\n')
            # mwfile.write('25\n')
            # mwfile.write('1\n')
            # mwfile.write('\n')
            # mwfile.write('\n')
            # mwfile.write('-P\n')
            # mwfile.write('\n')
            # mwfile.write('0\n')
            mwfile.write('-3\n')
            mwfile.write(f'{index}\n')
            mwfile.write('-1\n')
            mwfile.write('18\n')
            mwfile.write('1\n')
            mwfile.write(f'{self.inputfile}\n')
            mwfile.write(f'{multi}\n')
            mwfile.write(f'{nstate}\n')
            mwfile.write('\n')
            mwfile.write('1\n')
            if not self.cube_template:
                mwfile.write('3\n')
                # mwfile.write('0.1\n')
            else:
                mwfile.write('8\n')
                mwfile.write(f'{self.cube_template}\n')
            mwfile.write('10\n')
            mwfile.write('1\n')
            mwfile.write('11\n')
            mwfile.write('1\n')
            mwfile.write('0\n0\n0\nq\n')
    
    # def __read_soc(self):
    #     """
    #     读取SOC值.
    #     """
    #     soc = []
    #     with open(self.basename,'r') as orcaout:
    #         for line in orcaout:
    #     return soc

    def __run_multiwfn(self):
        Min = open(f'{self.basename}.minp', 'r')
        Mout = open(f'{self.basename}.mout', 'w')
        Merr = open('Multiwfn_err', 'w')
        try:
            p = subprocess.Popen(['Multiwfn', f'{self.basename}.molden'],
                        stdin=Min, stdout=subprocess.PIPE, text=True, stderr=Merr)
        except FileNotFoundError:
            p = subprocess.Popen(['Multiwfn.exe', f'{self.basename}.molden'],
                        stdin=Min, stdout=subprocess.PIPE, text=True, stderr=Merr)
        for l in p.stdout:
            if 'Progress' in l:
                l = l.strip('\n')
                print(l, end='\r')
            else:
                Mout.write(l)
        print('')
        return_code = p.wait()  # Wait for process to complete and get the return code
        if return_code != 0:
            print(f"Multiwfn exited with error code {return_code}. Check 'Multiwfn_err' for details.")
        Min.close()
        Mout.close()
        Merr.close()

    def calculate_he_cub(self):
        """
        计算每个重原子的空穴和电子密度.
        """
        multi_n2s={"1":"S","2":"D","3":"T","4":"Q","5":"P","6":"H","7":"O","8":"N","9":"E"}
        self.exlabel = []
        for multi,ns in [('1',self.siglet_sn),('3',self.triplet_sn)]:
            exlabel=f'{multi_n2s[multi]}{ns}'
            self.exlabel.append(exlabel)
            hole_file = os.path.join(f'{self.basename}', f'{self.basename}_{exlabel}_hole.cub')
            electron_file = os.path.join(f'{self.basename}', f'{self.basename}_{exlabel}_elec.cub')
            if not os.path.exists(hole_file) or not os.path.exists(electron_file):
                print(f'Generate Hole/Electron cube for Multiplicity:{multi} and excited state:{ns}...')
                self.__gen_multiwfn_input('0',multi,ns)
                self.__run_multiwfn()
                if not self.cube_template:
                    self.cube_template = hole_file
                shutil.move('hole.cub', hole_file)
                shutil.move('electron.cub', electron_file)
            else:
                print(f'\rHole/Electron cube for Multiplicity:{multi} and excited state:{ns} already exists.',end="")
                sys.stdout.flush()
        for i,index in enumerate(self.heavy_atom_index):
            for multi,ns in [('1',self.siglet_sn),('3',self.triplet_sn)]:
                exlabel=f'{multi_n2s[multi]}{ns}'
                hole_file = os.path.join(f'{self.basename}', f'{self.basename}_{index}_{exlabel}_hole.cub')
                electron_file = os.path.join(f'{self.basename}', f'{self.basename}_{index}_{exlabel}_elec.cub')
                mout_file = os.path.join(f'{self.basename}', f'{self.basename}_{index}_{exlabel}.mout')
                if not os.path.exists(hole_file) or not os.path.exists(electron_file):
                    print(f'Run Multiwfn for {self.heavy_atom_sn[i]}:{exlabel}...')
                    self.__gen_multiwfn_input(index,multi,ns)
                    self.__run_multiwfn()
                    shutil.move('hole.cub', hole_file)
                    shutil.move('electron.cub', electron_file)
                    shutil.move(f'{self.basename}.mout', mout_file)
                else:
                    print(f'\r{self.heavy_atom_sn[i]}:{exlabel} cubes already exists.',end='') 
                    sys.stdout.flush()
                self.df.loc[index,f'{exlabel}_hole_cub'] = f'{self.basename}_{index}_{exlabel}_hole.cub'
                self.df.loc[index,f'{exlabel}_elec_cub'] = f'{self.basename}_{index}_{exlabel}_elec.cub'

    def calc_cube(self):
        def vector_angle(vect1, vect2):
            # vect1 = np.array(vect1.split(','), dtype=float)
            # vect2 = np.array(vect2.split(','), dtype=float)
            dot_product = np.dot(vect1, vect2)
            norm1 = np.linalg.norm(vect1)
            norm2 = np.linalg.norm(vect2)
            cos_theta = dot_product / (norm1 * norm2)
            angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 限制值域防止浮点误差
            # 如果角度大于pi/2，则使用它的补角
            if angle_rad > np.pi / 2:
                angle_rad = np.pi - angle_rad
            return np.rad2deg(angle_rad)
        
        def int_rot(int1,int2,angle):
            return min(float(int1),float(int2))*angle

        def cross_prod(va, vb):
            va = np.array(va.split(','), dtype=float)
            vb = np.array(vb.split(','), dtype=float)
            dp = np.dot(va, vb)
            if dp < 0:
                vb = -vb
            cp =  ','.join([f'{i:.8e}' for i in np.cross(va, vb)])
            return cp

        def vect_add(va,vb):
            dp = np.dot(va, vb)
            if dp < 0:
                vb = -vb
            return va+vb
        
        def scale_vector(vect, scale):
            vect = np.array(vect.split(','), dtype=float)
            return scale * vect


        def contrib2sumvect(v1,sumvect):
            perc=np.dot(v1, sumvect)/np.dot(sumvect,sumvect)
            cos_angle = np.dot(v1, sumvect) / (np.linalg.norm(v1) * np.linalg.norm(sumvect))
            angle = np.arccos(cos_angle)  # 返回的是弧度
            return (perc,np.degrees(angle))  # 转换为度
        
        os.chdir(self.basename)
        cub_columns = [column for column in self.df.columns if column.endswith('cub')]
        # for column in cub_columns:
        #     self.df[new_col] = self.df[column].apply(self.__cub2stat)

        for column in cub_columns:
            new_col0 = column[:-3]+'int'
            new_col1 = column[:-3]+'main'
            new_col2 = column[:-3]+'sub1'
            # new_col3 = column[:-3]+'sub2'
            new_col4 = column[:-3]+'vect'
            self.df[new_col0] = self.df[column].apply(self.__cub2stat)
            self.df[new_col1] = self.df[column].apply(lambda x: self.__cub2svd(x)[0][0])
            self.df[new_col2] = self.df[column].apply(lambda x: self.__cub2svd(x)[0][1])
            self.df[new_col4] = self.df[column].apply(lambda x: self.__cub2svd(x)[1][0])
 
        for i,index in enumerate(self.heavy_atom_index):
            xyz = np.array([float(x)*0.529177249 for x in self.coords[int(index)-1]])
            self.df.loc[index,'coord'] = np.array2string(xyz,precision=5).strip('[ ]')
        hole_vect = [f'{i}_hole_vect' for i in self.exlabel]
        elec_vect = [f'{i}_elec_vect' for i in self.exlabel]
        hole_int = [f'{i}_hole_int' for i in self.exlabel]
        elec_int = [f'{i}_elec_int' for i in self.exlabel]
        hole_main = [f'{i}_hole_main' for i in self.exlabel]
        elec_main = [f'{i}_elec_main' for i in self.exlabel]
        hole_sub1 = [f'{i}_hole_sub1' for i in self.exlabel]
        elec_sub1 = [f'{i}_elec_sub1' for i in self.exlabel]
        hole_cub = [f'{i}_hole_cub' for i in self.exlabel]
        elec_cub = [f'{i}_elec_cub' for i in self.exlabel]

        if len(hole_vect) == 2 and len(elec_vect) == 2:
            # self.df['state1_len'] = self.df[hole_main[0]] - self.df[hole_sub1[0]] + self.df[elec_main[0]] - self.df[elec_sub1[0]]
            # self.df['state2_len'] = self.df[hole_main[1]] - self.df[hole_sub1[1]] + self.df[elec_main[1]] - self.df[elec_sub1[1]]
            # self.df['hole_vect_mult'] = self.df[hole_main[0]] * self.df[hole_main[1]]
            # self.df['elec_vect_mult'] = self.df.apply(lambda row: row[elec_main[0]] * row[elec_main[1]], axis=1)
            # self.df['elec_overlap'] = self.df.apply(lambda row: self.__cub_overlap(row[elec_cub[0]], row[elec_cub[1]]), axis=1)
            # self.df['hole_overlap'] = self.df.apply(lambda row: self.__cub_overlap(row[hole_cub[0]], row[hole_cub[1]]), axis=1)
            # self.df['hole_int'] = self.df[hole_int[0]] * self.df[hole_int[1]]
            # self.df['elec_int'] = self.df[elec_int[0]] * self.df[elec_int[1]]
            # S1态的空穴电子积分之和乘以T1态的空穴电子积分之和
            # self.df['int1'] = (self.df[hole_int[0]] + self.df[elec_int[0]]) * (self.df[hole_int[1]] + self.df[elec_int[1]])
            self.df[['state1_int','state1_cub']] = self.df.apply(lambda row: self.__cub_merge(row[hole_cub[0]], row[elec_cub[0]]), axis=1, result_type='expand')
            self.df[['state2_int','state2_cub']] = self.df.apply(lambda row: self.__cub_merge(row[hole_cub[1]], row[elec_cub[1]]), axis=1, result_type='expand')
            self.df['int1'] = self.df['state1_int'] * self.df['state2_int']
            self.df['int2'] = self.df.apply(lambda row: self.__cub_overlap(row['state1_cub'], row['state2_cub']), axis=1)
            # S1态的空穴电子积分之和乘以T1态的空穴电子积分之和
            # self.df['vect_len'] = (self.df[hole_main[0]] + self.df[elec_main[0]]) * (self.df[hole_main[1]] + self.df[elec_main[1]])
            # self.df['vect_len'] = self.df['state1_len'] * self.df['state2_len']
            
            #计算S1/T1态的vect 是hole和elec的主矢量的权重合
            self.df['hole_vect_len1'] = self.df[hole_main[0]]*self.df[hole_vect[0]]
            self.df['elec_vect_len1'] = self.df[elec_main[0]]*self.df[elec_vect[0]]
            self.df['hole_vect_len2'] = self.df[hole_main[1]]*self.df[hole_vect[1]]
            self.df['elec_vect_len2'] = self.df[elec_main[1]]*self.df[elec_vect[1]]
            self.df['vect_len1'] = self.df.apply(lambda row: vect_add(row['hole_vect_len1'],row['elec_vect_len1']), axis=1) 
            self.df['vect_len2'] = self.df.apply(lambda row: vect_add(row['hole_vect_len2'],row['elec_vect_len2']), axis=1)
            self.df['angle1_deg'] = self.df.apply(lambda row: vector_angle(row['vect_len1'], row['vect_len2']), axis=1)
            self.df['hole_vect_int1'] = self.df[hole_int[0]]*self.df[hole_vect[0]]
            self.df['elec_vect_int1'] = self.df[elec_int[0]]*self.df[elec_vect[0]]
            self.df['hole_vect_int2'] = self.df[hole_int[1]]*self.df[hole_vect[1]]
            self.df['elec_vect_int2'] = self.df[elec_int[1]]*self.df[elec_vect[1]]
            self.df['vect_int1'] = self.df.apply(lambda row: vect_add(row['hole_vect_int1'],row['elec_vect_int1']), axis=1) 
            self.df['vect_int2'] = self.df.apply(lambda row: vect_add(row['hole_vect_int2'],row['elec_vect_int2']), axis=1)
            self.df['angle2_deg'] = self.df.apply(lambda row: vector_angle(row['vect_int1'], row['vect_int2']), axis=1)
            self.df['vect_merge1']= self.df['state1_cub'].apply(self.__cub2svd).apply(lambda x: x[1][0])
            self.df['vect_merge2']= self.df['state2_cub'].apply(self.__cub2svd).apply(lambda x: x[1][0])
            self.df['angle3_deg'] = self.df.apply(lambda row: vector_angle(row['vect_merge1'], row['vect_merge2']), axis=1)
            # self.df['int_sin_rot'] = self.df['int'] * self.df['rot_angle'].apply(np.deg2rad).apply(np.sin)
            # self.df['int_sin_sum'] = self.df.groupby('symbol')['int_sin_rot'].transform('sum')

            self.df['int1_esoc'] = self.df['int1'] * self.df['symbol'].map(self.elem2esoc)
            self.df['int2_esoc'] = self.df['int2'] * self.df['symbol'].map(self.elem2esoc)
            # self.df['int3_esoc'] = self.df['int3'] * self.df['symbol'].map(self.elem2esoc)

            self.df['angle1_sin'] = self.df['angle1_deg'].apply(np.deg2rad).apply(np.sin)
            self.df['angle2_sin'] = self.df['angle2_deg'].apply(np.deg2rad).apply(np.sin)
            self.df['angle3_sin'] = self.df['angle3_deg'].apply(np.deg2rad).apply(np.sin)
            
            self.df['11_is'] = self.df['angle1_sin']*self.df['int1_esoc']
            self.df['12_is'] = self.df['angle2_sin']*self.df['int1_esoc']
            self.df['13_is'] = self.df['angle3_sin']*self.df['int1_esoc']
            self.df['21_is'] = self.df['angle1_sin']*self.df['int2_esoc']    
            self.df['22_is'] = self.df['angle2_sin']*self.df['int2_esoc']
            self.df['23_is'] = self.df['angle3_sin']*self.df['int2_esoc']
            
            self.df['int1_esoc_sum'] = self.df.groupby('symbol')['int1_esoc'].transform('sum')
            self.df['int2_esoc_sum'] = self.df.groupby('symbol')['int2_esoc'].transform('sum')
            self.df['11_is_sum'] = self.df.groupby('symbol')['11_is'].transform('sum')
            self.df['12_is_sum'] = self.df.groupby('symbol')['12_is'].transform('sum')
            self.df['13_is_sum'] = self.df.groupby('symbol')['13_is'].transform('sum')
            self.df['21_is_sum'] = self.df.groupby('symbol')['21_is'].transform('sum')
            self.df['22_is_sum'] = self.df.groupby('symbol')['22_is'].transform('sum')
            self.df['23_is_sum'] = self.df.groupby('symbol')['23_is'].transform('sum')
            #计算S1/T1态的int 是hole和elec的主矢量的权重合，轨道积分值为权重
            # self.df['state1_int_vect'] = self.df[hole_int[0]]*self.df[hole_vect[0]] + self.df[elec_int[0]]*self.df[elec_vect[0]]
            # self.df['state2_int_vect'] = self.df[hole_int[1]]*self.df[hole_vect[1]] + self.df[elec_int[1]]*self.df[elec_vect[1]]

            # self.df['int_cross'] = self.df.apply(lambda row: np.cross(row['state1_int_vect'], row['state2_int_vect']), axis=1)
            # self.df['len_cross'] = self.df.apply(lambda row: np.cross(row['state1_len_vect'], row['state2_len_vect']), axis=1)

            # self.df['int_cross_esoc'] = self.df['int_cross'] * self.df['symbol'].map(self.elem2esoc)  
            # self.df['len_cross_esoc'] = self.df['len_cross'] * self.df['symbol'].map(self.elem2esoc)

            # self.df['int_scalar'] = self.df['int_cross'].apply(np.linalg.norm)
            # self.df['len_scalar'] = self.df['len_cross'].apply(np.linalg.norm)
            # self.df['int_scalar_esoc'] = self.df['int_scalar'] * self.df['symbol'].map(self.elem2esoc)
            # self.df['len_scalar_esoc'] = self.df['len_scalar'] * self.df['symbol'].map(self.elem2esoc)

            # self.df['int_cross_sum']=self.df.groupby('symbol')['int_cross'].transform('sum')
            # self.df['len_cross_sum']=self.df.groupby('symbol')['len_cross'].transform('sum')    
            # self.df['vect_len_sum']=self.df.groupby('symbol')['vect_len'].transform('sum')  
            # self.df['int_sum']=self.df.groupby('symbol')['int'].transform('sum')
            # # self.df['int_cross_esoc_sum']=self.df.groupby('symbol')['int_cross_esoc'].transform('sum')
            # self.df['len_cross_esoc_sum']=self.df.groupby('symbol')['len_cross_esoc'].transform('sum')
            # # self.df['int_cross_esoc_sum_norm']=self.df['int_cross_esoc_sum'].apply(np.linalg.norm)
            # self.df['len_cross_esoc_sum_norm']=self.df['len_cross_esoc_sum'].apply(np.linalg.norm)


            # # self.df['int_scalar_sum']=self.df.groupby('symbol')['int_scalar'].transform('sum')
            # self.df['len_scalar_sum']=self.df.groupby('symbol')['len_scalar'].transform('sum')
            # # self.df['int_scalar_esoc_sum']=self.df.groupby('symbol')['int_scalar_esoc'].transform('sum')
            # self.df['len_scalar_esoc_sum']=self.df.groupby('symbol')['len_scalar_esoc'].transform('sum')

        
            # 计算hole和electron矢量的cross product
            # self.df['hole_cross'] = self.df.apply(lambda row: np.cross(row[hole_vect[0]], row[hole_vect[1]]), axis=1)
            # self.df['elec_cross'] = self.df.apply(lambda row: np.cross(row[elec_vect[0]], row[elec_vect[1]]), axis=1)
            # self.df['elec_cross'] = self.df.apply(lambda row: cross_prod(row[elec_vect[0]], row[elec_vect[1]]), axis=1)
            # 计算hole/elec列矢量的夹角
            # self.df['hole_angle_rot'] = self.df.apply(lambda row: vector_angle(row[hole_vect[0]], row[hole_vect[1]]), axis=1)
            # self.df['elec_angle_rot'] = self.df.apply(lambda row: vector_angle(row[elec_vect[0]], row[elec_vect[1]]), axis=1)
            # self.df['sin_hole_angle'] = np.sin(np.radians(self.df['hole_angle_rot']))
            # self.df['sin_elec_angle'] = np.sin(np.radians(self.df['elec_angle_rot']))
            # self.df["O_Hscalar"] = self.df['sin_hole_angle'] * self.df['hole_overlap'] * self.df['symbol'].map(self.elem2esoc)
            # self.df["L_Hscalar"] = self.df['sin_hole_angle'] * self.df['hole_vect_mult'] * self.df['symbol'].map(self.elem2esoc)
            # self.df["I_Hscalar"] = self.df['sin_hole_angle'] * self.df['hole_int'] * self.df['symbol'].map(self.elem2esoc)
            # self.df["O_Escalar"] = self.df['sin_elec_angle'] * self.df['elec_overlap'] * self.df['symbol'].map(self.elem2esoc)
            # self.df["L_Escalar"] = self.df['sin_elec_angle'] * self.df['elec_vect_mult'] * self.df['symbol'].map(self.elem2esoc)
            # self.df["I_Escalar"] = self.df['sin_elec_angle'] * self.df['elec_int'] * self.df['symbol'].map(self.elem2esoc)
            # self.df['O_Hcross'] = self.df['hole_cross'] * self.df['hole_overlap'] * self.df['symbol'].map(self.elem2esoc)
            # self.df['L_Hcross'] = self.df['hole_cross'] * self.df['hole_vect_mult'] * self.df['symbol'].map(self.elem2esoc)
            # self.df['I_Hcross'] = self.df['hole_cross'] * self.df['hole_int'] * self.df['symbol'].map(self.elem2esoc)
            # self.df['O_Ecross'] = self.df['elec_cross'] * self.df['elec_overlap']  * self.df['symbol'].map(self.elem2esoc)
            # self.df['L_Ecross'] = self.df['elec_cross'] * self.df['elec_vect_mult'] * self.df['symbol'].map(self.elem2esoc)
            # self.df['I_Ecross'] = self.df['elec_cross'] * self.df['elec_int'] * self.df['symbol'].map(self.elem2esoc)
            # self.df['O_Hscalar_sum']=self.df.groupby('symbol')['O_Hscalar'].transform('sum')
            # self.df['L_Hscalar_sum']=self.df.groupby('symbol')['L_Hscalar'].transform('sum')
            # self.df['I_Hscalar_sum']=self.df.groupby('symbol')['I_Hscalar'].transform('sum')
            # self.df['O_Escalar_sum']=self.df.groupby('symbol')['O_Escalar'].transform('sum')
            # self.df['L_Escalar_sum']=self.df.groupby('symbol')['L_Escalar'].transform('sum')
            # self.df['I_Escalar_sum']=self.df.groupby('symbol')['I_Escalar'].transform('sum')
            # self.df['O_scalar_sum']=self.df['O_Hscalar_sum']+self.df['O_Escalar_sum']
            # self.df['L_scalar_sum']=self.df['L_Hscalar_sum']+self.df['L_Escalar_sum']
            # self.df['I_scalar_sum']=self.df['I_Hscalar_sum']+self.df['I_Escalar_sum']
            # self.df['O_Hcross_sum']=self.df.groupby('symbol')['O_Hcross'].transform('sum')
            # self.df['L_Hcross_sum']=self.df.groupby('symbol')['L_Hcross'].transform('sum')
            # self.df['I_Hcross_sum']=self.df.groupby('symbol')['I_Hcross'].transform('sum')
            # self.df['O_Ecross_sum']=self.df.groupby('symbol')['O_Ecross'].transform('sum')
            # self.df['L_Ecross_sum']=self.df.groupby('symbol')['L_Ecross'].transform('sum')
            # self.df['I_Ecross_sum']=self.df.groupby('symbol')['I_Ecross'].transform('sum')
            # self.df['O_cross_sum']=self.df['O_Hcross_sum']+self.df['O_Ecross_sum']
            # self.df['L_cross_sum']=self.df['L_Hcross_sum']+self.df['L_Ecross_sum']
            # self.df['I_cross_sum']=self.df['I_Hcross_sum']+self.df['I_Ecross_sum']

            # self.df['hole_cross_sum'] = [np.sum(self.df['hole_cross'].values, axis=0)] * len(self.df)

            # self.df['elec_cross_sum'] = [np.sum(self.df['elec_cross'].values, axis=0)] * len(self.df)
            # self.df['hole_cross_norm'] = np.linalg.norm(self.df['hole_cross_sum'].iloc[0])
            # self.df['elec_cross_norm'] = np.linalg.norm(self.df['elec_cross_sum'].iloc[0])
            # self.df[['hole_cross_contrib','hole_cross_angle']] = self.df.apply(lambda row: contrib2sumvect(row['hole_cross'],row['hole_cross_sum']), axis=1,result_type='expand')
        else:
            print("Error: Did not find exactly two 'hole' or 'elec' columns.")
            sys.exit()
        # if len(hole_int) == 2 and len(elec_int) == 2:
        #     self.df['hole_rot'] = self.df.apply(lambda row: int_rot(row[hole_int[0]], row[hole_int[1]], row['hole_angle_rot']), axis=1)
        #     self.df['elec_rot'] = self.df.apply(lambda row: int_rot(row[elec_int[0]], row[elec_int[1]], row['elec_angle_rot']), axis=1)
        #     self.df['rot'] = self.df['hole_rot'] + self.df['elec_rot']
        # else:
        #     print("Error: Did not find exactly two 'hole_int' or 'elec_int' columns")
        prt_columns = [column for column in self.df.columns if column.split('_')[-1] in ['esoc','sin','is','sum']]
        print()
        # print(self.df[prt_columns].T)
        vect_columns = [column for column in self.df.columns if column.endswith('vect')]
        print(f"commands for plotting using vcube save in {self.basename}_arrow.txt")
        file = open(f'{self.basename}_arrow.txt','w')
        for col in vect_columns:
            for index,row in enumerate(self.df[col]):
                sn = self.df['sn'].iloc[index]
                file.write(f"{col}_{sn}:  varrow {row} -o {self.df['coord'].iloc[index]} -s 5\n")
        if os.path.exists(f'../{self.basename}_arrow.txt'):
            os.remove(f'../{self.basename}_arrow.txt')
        shutil.move(f'{self.basename}_arrow.txt','../')
        os.chdir('..')
        self.df.to_csv(f'{self.basename}_alldata.csv')
        sum_cols = ['symbol'] + [column for column in self.df.columns if column.endswith(('sum','is','esoc','sin','z'))]
        # 首先，筛选出'z'列大于18的行
        filtered_df = self.df[self.df['z'] < 18]
        unique_df = filtered_df.drop_duplicates(subset='symbol').reset_index(drop=True)
        final_df = pd.concat([self.df[self.df['z'] >= 18],unique_df]).reset_index(drop=True)
        final_df.to_csv(f'{self.basename}_sumdata.csv')
        self.sumdata = final_df

    def __read_cub(self, cubefile):
        with open(cubefile, 'r') as file:
            # 跳过标题两行
            next(file)
            next(file)
            # 读取原子数量和原点坐标
            atom_count, origin_x, origin_y, origin_z = map(float, file.readline().split())
            # 读取三个平移矢量方向上的数据点数和矢量分量
            nx, vx_x, vx_y, vx_z = map(float, file.readline().split())
            ny, vy_x, vy_y, vy_z = map(float, file.readline().split())
            nz, vz_x, vz_y, vz_z = map(float, file.readline().split())
            # 读取格点数据
            for _ in range(int(atom_count)):
                file.readline()
            data = []
            for line in file:
                values = line.strip().split()
                data.extend(map(float, values))
            # 格点数据转换为NumPy数组
            data = np.array(data)
            # 生成格点坐标
            grid_points = np.zeros((int(nx * ny * nz), 4))
            index = 0
            for i in range(int(nx)):
                for j in range(int(ny)):
                    for k in range(int(nz)):
                        x = origin_x + i * vx_x + j * vy_x + k * vz_x
                        y = origin_y + i * vx_y + j * vy_y + k * vz_y
                        z = origin_z + i * vx_z + j * vy_z + k * vz_z
                        grid_points[index, :3] = [x, y, z]
                        grid_points[index, 3] = data[index]
                        index += 1
            return grid_points

    def __cub2svd(self,cubefile):
        sys.stdout.flush()
        basename=os.path.splitext(cubefile)[0]
        ufile = f'{basename}_U.npy'
        vfile = f'{basename}_Vt.npy'
        sfile = f'{basename}_S.npy'
        if os.path.exists(ufile) and os.path.exists(vfile) and os.path.exists(sfile):
            print(f'\rSVD for {cubefile} already exists.',end="")
            sys.stdout.flush()
            U=np.load(ufile)
            S=np.load(sfile)
            Vt=np.load(vfile)
        else:
            data = self.__read_cub(cubefile)
            coordinates = data[:, :3]
            # 分离坐标和权重
            weights = data[:, 3]
            # 根据权重计算加权均值
            weighted_mean = np.average(coordinates, axis=0, weights=weights)
            # 计算加权零均值坐标
            zero_mean_coords = coordinates - weighted_mean
            # 构造加权协方差矩阵
            weighted_cov_matrix = zero_mean_coords.T @ (zero_mean_coords * weights[:, np.newaxis])
            # 执行SVD分解
            U, S, Vt = np.linalg.svd(weighted_cov_matrix)
            np.save(ufile, U)
            np.save(vfile, Vt)
            np.save(sfile, S)
            sstr=" ".join([f'{i:.3f}' for i in S])
            print(f'\rCalculated SVD for {cubefile}...{sstr}',end="")
            sys.stdout.flush()
        return S,Vt

    def __cub_overlap(self,cubefile1,cubefile2):
        outname = os.path.splitext(cubefile1)[0] + '_' + os.path.splitext(cubefile2)[0] + '_overlap.txt'
        if not os.path.exists(outname):
            with open('overlap.minp', 'w') as mwfile:
                mwfile.write('13\n')
                mwfile.write('11\n')
                mwfile.write('6\n')
                mwfile.write(f'{cubefile2}\n')
                mwfile.write('17\n')
                mwfile.write('1\n')
                mwfile.write('-1\n')
                mwfile.write('q\n')
            Minp = open('overlap.minp', 'r')
            Mout = open(outname, 'w')
            Merr = open('overlap.merr', 'w')
            try:
                p = subprocess.Popen(['Multiwfn', cubefile1],
                            stdin=Minp, stdout=subprocess.PIPE, text=True, stderr=Merr)
            except FileNotFoundError:
                p = subprocess.Popen(['Multiwfn.exe', f'{self.basename}.molden'],
                            stdin=Minp, stdout=subprocess.PIPE, text=True, stderr=Merr)
            read_flag = False
            for l in p.stdout:
                if 'Progress' in l:
                    pass
                    # l = l.strip('\n')
                    # print(l, end='\r')
                else:
                    Mout.write(l)
            # print('')
            return_code = p.wait()  # Wait for process to complete and get the return code
            if return_code != 0:
                print(f"Multiwfn exited with error code {return_code}. Check 'overlap.merr' for details.")
            Minp.close()
            Merr.close()
            Mout.close()
        overlap_out=open(outname,'r')
        for l in overlap_out:
            if  "Integral of positive data:" in l:
                overlap=float(l.strip().split()[-1])
                break
        print(f'\rCalculation overlap between {cubefile1} and {cubefile2}...{overlap:.10f}',end="") 
        sys.stdout.flush()
        return overlap

    def __cub_merge(self,cubefile1,cubefile2):
        outname='_'.join(os.path.splitext(cubefile1)[0].split('_')[:-1])
        outcube = outname + '.cub'
        outtxt = outname + '.txt'
        if not os.path.exists(outcube) or not os.path.exists(outtxt):
            with open('cube_merge.minp', 'w') as mwfile:
                mwfile.write('13\n')
                mwfile.write('11\n')
                mwfile.write('2\n')
                mwfile.write(f'{cubefile2}\n')
                mwfile.write('17\n')
                mwfile.write('1\n')
                mwfile.write('0\n')
                mwfile.write(f'{outcube}\n')
                mwfile.write('-1\n')
                mwfile.write('q\n')
            Minp = open('cube_merge.minp', 'r')
            Mout = open(outtxt, 'w')
            Merr = open('cube_merge.merr', 'w')
            try:
                p = subprocess.Popen(['Multiwfn', cubefile1],
                            stdin=Minp, stdout=subprocess.PIPE, text=True, stderr=Merr)
            except FileNotFoundError:
                p = subprocess.Popen(['Multiwfn.exe', f'{self.basename}.molden'],
                            stdin=Minp, stdout=subprocess.PIPE, text=True, stderr=Merr)
            for l in p.stdout:
                if 'Progress' in l:
                    pass
                    # l = l.strip('\n')
                    # print(l, end='\r')
                else:
                    Mout.write(l)
            # print('')
            return_code = p.wait()  # Wait for process to complete and get the return code
            if return_code != 0:
                print(f"Multiwfn exited with error code {return_code}. Check 'overlap.merr' for details.")
            Minp.close()
            Merr.close()
            Mout.close()
        overlap_out=open(outtxt,'r')
        for l in overlap_out:
            if  "Integral of positive data:" in l:
                integral=float(l.strip().split()[-1])
                break
        print(f'\rCalculation integral of merged {cubefile1} and {cubefile2}...{integral:.10f}',end="") 
        sys.stdout.flush()
        return integral,outcube


    def __cub2stat(self,cubefile):
        basename=os.path.splitext(cubefile)[0]
        statfile = f'{basename}_stat.txt'
        need_recalc=True
        if os.path.exists(statfile):
            print(f'\rStat for {cubefile} already exists.',end='')
            sys.stdout.flush()
            with open(statfile,'r') as f:
                integral = f.readline().strip()
                if not integral:
                    need_recalc=True
                else:
                    need_recalc=False
        if need_recalc:
            with open('cub_int.minp', 'w') as mwfile:
                mwfile.write('13\n')
                mwfile.write('17\n')
                mwfile.write('1\n')
                mwfile.write('-1\n')
                mwfile.write('q\n')
            command = "cat cub_int.minp | Multiwfn {:s} > cub_int_tmp.mout".format(cubefile)
            result = subprocess.run(command, shell=True, text=True, capture_output=True)
            if result.returncode == 0:
                pass
                #print("Finished Sucessful")
            else:
                print(f"Something wrong：{result.returncode}")
                print(result.stderr)
            for line in open('cub_int_tmp.mout'):
            # 使用正则表达式搜索冒号后面的数字
                match = re.search(r'Integral of all data:\s*([0-9.]+)', line)
            # 如果匹配成功，打印出数字
                if match:
                    integral = match.group(1)
                    break
            with open(statfile,'w') as f:
                f.write(integral)
            print(f'\rCalculation intergral of {cubefile}...{float(integral):.10f}',end="")
            sys.stdout.flush()
        return float(integral)
    

alldf=pd.DataFrame()
for outfile in args.inputfile:
    haoroc=HeavyAtomOrbRotCompos(outfile,args.heavy,args.st)
    df = haoroc.sumdata
    print(df[[i for i in df.columns if i.endswith(('sum','angle','deg'))]])
    df_long = df.melt(id_vars=['symbol','sn'], var_name='data', value_name='value',value_vars=[i for i in df.columns if i.endswith(('sum','is','esoc','sin'))])  
    df_long['combi_col'] = df_long['sn'] +'_'+df_long['data']
    new_df = pd.DataFrame([df_long['value'].values], columns=df_long['combi_col'].values)
    grouped_columns = defaultdict(list)
    pattern = re.compile(r'([A-Z][a-z]?)(\d+)_(\d+_is_sum|int[12]_esoc_sum)')
    for col in new_df.columns:
        match = pattern.match(col)
        if match:
            key = match.group(1)+'_'+match.group(3)
            grouped_columns[key].append(col)
    columns_to_remove = [col for cols in grouped_columns.values() for col in cols[1:]]
    new_df = new_df.drop(columns=columns_to_remove)
    columns_mapping = {v[0]:k for k,v in grouped_columns.items()}
    new_df = new_df.rename(columns=columns_mapping)
    cols2retain = [i for i in new_df.columns if haoroc.elem2an[''.join([j for j in i.split('_')[0] if not j.isdigit()])] > 18 or i.endswith('is_sum')] + ['system']
    new_df['system'] = os.path.splitext(os.path.basename(outfile))[0]
    alldf=pd.concat([alldf,new_df[cols2retain]],axis=0)
alldf.set_index('system',inplace=True)
alldf.to_csv('alldata.csv')
x11=alldf[[col for col in alldf.columns if col.split('_')[1] == '11' and col.endswith('is_sum')]].sum(axis=1).rename('x11')
x12=alldf[[col for col in alldf.columns if col.split('_')[1] == '12' and col.endswith('is_sum')]].sum(axis=1).rename('x12')
x13=alldf[[col for col in alldf.columns if col.split('_')[1] == '13' and col.endswith('is_sum')]].sum(axis=1).rename('x13')
x21=alldf[[col for col in alldf.columns if col.split('_')[1] == '21' and col.endswith('is_sum')]].sum(axis=1).rename('x21')
x22=alldf[[col for col in alldf.columns if col.split('_')[1] == '22' and col.endswith('is_sum')]].sum(axis=1).rename('x22')
x23=alldf[[col for col in alldf.columns if col.split('_')[1] == '23' and col.endswith('is_sum')]].sum(axis=1).rename('x23')
b11=alldf[[col for col in alldf.columns if col.split('_')[1] == '11' and col.endswith('is_sum') and col.startswith('Br')]].sum(axis=1).rename('b11')
b12=alldf[[col for col in alldf.columns if col.split('_')[1] == '12' and col.endswith('is_sum') and col.startswith('Br')]].sum(axis=1).rename('b12')
b13=alldf[[col for col in alldf.columns if col.split('_')[1] == '13' and col.endswith('is_sum') and col.startswith('Br')]].sum(axis=1).rename('b13')
b21=alldf[[col for col in alldf.columns if col.split('_')[1] == '23' and col.endswith('is_sum') and col.startswith('Br')]].sum(axis=1).rename('b21')
b22=alldf[[col for col in alldf.columns if col.split('_')[1] == '22' and col.endswith('is_sum') and col.startswith('Br')]].sum(axis=1).rename('b22')
b23=alldf[[col for col in alldf.columns if col.split('_')[1] == '23' and col.endswith('is_sum') and col.startswith('Br')]].sum(axis=1).rename('b23')
# df_soc = pd.read_csv('bmat-soc1.dat', sep='\s+', index_col=0,names=['soc'],header=None)
df_join =  pd.concat([x11,x12,x13,x21,x22,x23,b11,b13,b23,b23],axis=1)
print(df_join)
# plt.scatter(df_join['x22'],df_join['soc'],label='b11')
# plt.show()
df_join.to_csv('sumdata.csv')
print("results saved in alldata.csv and sumdata.csv")
sys.exit()
