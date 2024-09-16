'''
Author: Fujin Wang
Date: 2024.05
Github: https://github.com/wang-fujin

Description:
该代码用于读取同济大学公开的电池数据集，方便后续预处理和分析。
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pprint
import os
import pickle
import functools
from scipy import interpolate


def interpolate_resample(resample=True, num_points=128):
    '''
    插值重采样装饰器,如果resample为True，那么就进行插值重采样，点数为num_points,默认为128；
    否则就不进行重采样
    :param resample: bool: 是否进行重采样
    :param num_points: int: 重采样的点数
    :return:
    '''
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self,*args, **kwargs):
            data = func(self,*args, **kwargs)
            new_df = pd.DataFrame()

            if resample:
                x = np.linspace(0, 1, data.shape[0])
                new_x = np.linspace(0, 1, num_points)
                for k in data:
                    if k == 'Status':
                        continue
                    try:
                        f1 = interpolate.interp1d(x, data[k], kind='linear')
                        new_df[k] = f1(new_x)
                    except ValueError:
                        print(data.shape[0])
            return new_df
        return wrapper
    return decorator


class Battery:
    def __init__(self,path='../Dataset_1_NCA_battery/CY25-1_1-#1.csv'):
        self.path = path
        self.df = pd.read_csv(path)
        file_name = path.split('/')[-1]
        self.temperature = int(file_name[2:4])
        self.charge_c_rate = file_name.split('-')[1].split('_')[0]
        self.discharge_c_rate = file_name.split('-')[1].split('_')[1]
        self.battery_id = file_name.split('#')[-1].split('.')[0]
        self.cycle_index = self._get_cycle_index()
        self.cycle_life = len(self.cycle_index)
        self.dropcycle = []
        print('-'*40,f' Battery #{self.battery_id} ','-'*40)
        print('电池寿命：',self.cycle_life)
        print('实验温度：',self.temperature)
        print('充电倍率：',self.charge_c_rate)
        print('放电倍率：',self.discharge_c_rate)
        print('变量名：',list(self.df.columns))
        print('-'*100)

    def _get_cycle_index(self):
        cycle_num = np.unique(self.df['cycle number'].values)
        return cycle_num

    def _check(self,cycle=None,variable=None):
        '''
        检查输入的cycle和variable是否合法
        :param cycle: int: 循环次数
        :param variable: str: 变量名
        :return: bool: 是否合法
        '''
        if cycle is not None:
            if cycle not in self.cycle_index:
                raise ValueError('cycle should be in [{},{}]'.format(int(self.cycle_index.min()),int(self.cycle_index.max())))
        if variable is not None:
            if variable not in self.df.columns:
                raise ValueError('variable should be in {}'.format(list(self.df.columns)))
        return True

    def get_cycle(self,cycle):
        '''
        获取第cycle次循环的数据
        :param cycle: int: 循环次数
        :return: DataFrame: 第cycle次循环的数据, columns:['time/s', 'control/V/mA', 'Ecell/V', '<I>/mA', 'Q discharge/mA.h', 'Q charge/mA.h', 'control/V', 'control/mA', 'cycle number']
        '''
        self._check(cycle=cycle)
        cycle_df = self.df[self.df['cycle number']==cycle]
        return cycle_df

    def get_degradation_trajectory(self):
        '''
        获取电池的容量退化轨迹
        :return:
        '''
        capacity = []
        for cycle in self.cycle_index:
            cycle_df = self.get_cycle(cycle)
            capacity.append(cycle_df['Q discharge/mA.h'].max())
        return capacity

    def get_value(self,cycle,variable):
        '''
        获取第cycle次循环的variable变量的值
        :param cycle: int: 循环次数
        :param variable: str: 变量名
        :return: series: 第cycle次循环的variable变量的值
        '''
        self._check(cycle=cycle,variable=variable)
        cycle_df = self.get_cycle(cycle)
        return cycle_df[variable].values

    def get_charge_stage(self,cycle):
        '''
        获取第cycle次循环的CCCV阶段的数据
        :param cycle: int: 循环次数
        :return: DataFrame: 第cycle次循环的CCCV阶段的数据, columns:['time/s', 'control/V/mA', 'Ecell/V', '<I>/mA', 'Q discharge/mA.h', 'Q charge/mA.h', 'control/V', 'control/mA', 'cycle number']
        '''
        self._check(cycle=cycle)
        cycle_df = self.get_cycle(cycle)
        charge_df = cycle_df[cycle_df['control/V/mA']>0]
        return charge_df

    @interpolate_resample(resample=True, num_points=219)
    def get_CC_stage(self,cycle,voltage_range=None):
        '''
        获取第cycle次循环的CC阶段的数据
        :param cycle: int: 循环次数
        :param voltage_range: list: 电压范围
        :return: DataFrame: 第cycle次循环的CC阶段的数据, columns:['time/s', 'control/V/mA', 'Ecell/V', '<I>/mA', 'Q discharge/mA.h', 'Q charge/mA.h', 'control/V', 'control/mA', 'cycle number']
        '''
        self._check(cycle=cycle)
        cycle_df = self.get_cycle(cycle)
        CC_df = cycle_df[cycle_df['control/mA']>0]

        if voltage_range is not None:
            CC_df = CC_df[CC_df['Ecell/V'].between(voltage_range[0],voltage_range[1])]
        return CC_df

    def get_CV_stage(self,cycle,current_range=None):
        '''
        获取第cycle次循环的CV阶段的数据
        :param cycle: int: 循环次数
        :param current_range: list: 电流范围
        :return: DataFrame: 第cycle次循环的CV阶段的数据, columns:['time/s', 'control/V/mA', 'Ecell/V', '<I>/mA', 'Q discharge/mA.h', 'Q charge/mA.h', 'control/V', 'control/mA', 'cycle number']
        '''
        self._check(cycle=cycle)
        cycle_df = self.get_cycle(cycle)
        CV_df = cycle_df[cycle_df['control/V']>0]

        if current_range is not None:
            CV_df = CV_df[CV_df['<I>/mA'].between(np.min(current_range),np.max(current_range))]
        return CV_df


    # @interpolate_resample(resample=True, num_points=470)  # NCA-batch2
    # @interpolate_resample(resample=True, num_points=548)  # NCA-batch1
    # @interpolate_resample(resample=True, num_points=518)  # NCA-batch3
    # @interpolate_resample(resample=True, num_points=617)  # NCA-batch4

    # @interpolate_resample(resample=True, num_points=544)  # NCM-batch1
    # @interpolate_resample(resample=True, num_points=550)  # NCM-batch2
    # @interpolate_resample(resample=True, num_points=508)  # NCM-batch3
    # @interpolate_resample(resample=True, num_points=538)  # NCA-total
    # @interpolate_resample(resample=True, num_points=534)  # NCM-total
    @interpolate_resample(resample=True, num_points=1102)  # NCM_NCA-total
    def get_CCCV_stage(self, cycle):
        charge_df = self.get_charge_stage(cycle)
        CCCV_df = charge_df[charge_df['<I>/mA'] > 1]

        diff = np.diff(CCCV_df.index)

        continuous_index = np.where(diff != 1)[0]

        if len(continuous_index) > 0:
            start_index = CCCV_df.index[continuous_index[0] + 1]
            CCCV_df = CCCV_df.loc[start_index:]

        if CCCV_df.shape[0] < 50:
            print(f"cycle: {cycle}")
            print('len is: ', CCCV_df.shape[0])
            return pd.DataFrame()
        return CCCV_df

    def plot_one_cycle_charge(self,cycle):
        '''
        绘制第cycle次循环的charge阶段的曲线
        :param cycle: int: 循环次数
        :return: None
        '''
        self._check(cycle=cycle)
        charge_df = self.get_charge_stage(cycle)


        fig, ax = plt.subplots(3, 1, figsize=(5, 5), dpi=200)
        ax[0].plot(charge_df['time/s'].values, charge_df['Ecell/V'].values, color='b', linewidth=2)
        ax[0].set_ylabel('Voltage/V')
        ax[0].axvline(x=charge_df['time/s'].values[-1], color='g', linestyle='--')

        ax[1].plot(charge_df['time/s'].values, charge_df['<I>/mA'].values, color='b', linewidth=2)
        ax[1].axvline(x=charge_df['time/s'].values[-1], color='g', linestyle='--')
        ax[1].set_ylabel('Current/mA')

        ax[2].plot(charge_df['time/s'].values, charge_df['Q charge/mA.h'].values, color='b', linewidth=2)
        ax[2].axvline(x=charge_df['time/s'].values[-1], color='g', linestyle='--')
        ax[2].set_ylabel('Charge Q/mA.h')
        ax[2].set_xlabel('Time/s')
        plt.tight_layout()
        plt.show()


    def plot_one_cycle_CCCV(self,cycle):
        '''
        绘制第cycle次循环的CCCV阶段的曲线
        :param cycle: int: 循环次数
        :return: None
        '''
        self._check(cycle=cycle)
        CC_df = self.get_CC_stage(cycle,voltage_range=[4.0,4.2])
        CV_df = self.get_CV_stage(cycle,current_range=[2000,1000])
        CCCV_df = self.get_CCCV_stage(cycle)

        fig, ax = plt.subplots(3, 1, figsize=(5, 5), dpi=200)
        ax[0].plot(CC_df['time/s'].values, CC_df['Ecell/V'].values, color='b', linewidth=2)
        ax[0].plot(CV_df['time/s'].values, CV_df['Ecell/V'].values, color='r', linewidth=2)
        ax[0].set_ylabel('Voltage/V')
        ax[0].axvline(x=CC_df['time/s'].values[-1], color='g', linestyle='--')

        ax[1].plot(CC_df['time/s'].values, CC_df['<I>/mA'].values, color='b', linewidth=2)
        ax[1].plot(CV_df['time/s'].values, CV_df['<I>/mA'].values, color='r', linewidth=2)
        ax[1].axvline(x=CC_df['time/s'].values[-1], color='g', linestyle='--')
        ax[1].set_ylabel('Current/mA')

        ax[2].plot(CC_df['time/s'].values, CC_df['Q charge/mA.h'].values, color='b', linewidth=2)
        ax[2].plot(CV_df['time/s'].values, CV_df['Q charge/mA.h'].values, color='r', linewidth=2)
        ax[2].axvline(x=CC_df['time/s'].values[-1], color='g', linestyle='--')
        ax[2].set_ylabel('Charge Q/mA.h')
        ax[2].set_xlabel('Time/s')
        plt.tight_layout()
        plt.show()

    def plot_one_cycle(self,cycle):
        '''
        绘制第cycle次循环的变量的曲线
        :param cycle: int: 循环次数
        :return: None
        '''
        self._check(cycle=cycle)
        cycle_df = self.get_cycle(cycle)

        time = cycle_df['time/s'].values
        voltage = cycle_df['Ecell/V'].values
        current = cycle_df['<I>/mA'].values
        discharge_capacity = cycle_df['Q discharge/mA.h'].values
        charge_capacity = cycle_df['Q charge/mA.h'].values

        fig,ax = plt.subplots(4,1,figsize=(6,6),dpi=200)
        ax[0].plot(time,voltage,color='b', linewidth=2)
        ax[0].set_ylabel('Voltage/V')
        ax[1].plot(time,current,color='r', linewidth=2)
        ax[1].set_ylabel('Current/mA')
        ax[2].plot(time,discharge_capacity,color='g', linewidth=2)
        ax[2].set_ylabel('Q Discharge/mA.h')
        ax[3].plot(time,charge_capacity,color='y', linewidth=2)
        ax[3].set_ylabel('Q Charge/mA.h')
        ax[3].set_xlabel('Time/s')
        plt.suptitle(f'Battery {self.battery_id} Cycle {cycle}')
        plt.tight_layout()
        plt.show()


def save_all_battery2pkl(filepath, all_battery):
    # Note that all_battery is a dictionary contains different Battery.
    with open(filepath, 'wb') as fp:
        pickle.dump(all_battery, fp)
    print("file to pkl finished!")



if __name__ == '__main__':
    # 一个简单的例子
    folder_path = r'data/tongji/Dataset_3_NCM_NCA_battery/'
    save_path = 'data/tongji/result/'

    all_bat_dict = {}
    battery_average_len = []
    key_name = ['Ecell/V','<I>/mA', 'Q discharge/mA.h', 'Q charge/mA.h']

    bat_idx = 1
    NCA_capacity_batch1 = 3273
    NCA_capacity_batch2 = 3364
    NCA_capacity_batch3 = 3306
    NCA_capacity_batch4 = 3269
    NCA_total_capacity = 3364

    NCM_capacity_batch1 = 3243
    NCM_capacity_batch2 = 3266
    NCM_capacity_batch3 = 3349
    NCM_total_capacity = 3349
    NCM_NCA_total_capacity = 2502
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        battery = Battery(filepath)

        print(filename)

        one_bat_dict = {}
        temp_bat_dict = {}
        one_battery_len = []
        one_bat_dict['summary'] = []

        wrong_label_list = []
        # 下面的代码用于处理标签异常的cycle.
        for index in range(1, battery.cycle_life + 1):
            # 处理一些下标错误的cycle_index
            if int(battery.cycle_index.min()) > 1:
                if 0.66 < ((battery.get_cycle(index + 1)['Q discharge/mA.h'].max())/NCM_NCA_total_capacity < 1.05
                          and not battery.get_CCCV_stage(index + 1).empty):
                    one_bat_dict['summary'].append((battery.get_cycle(index + 1)['Q discharge/mA.h'].max())/NCM_NCA_total_capacity)
                else:
                    wrong_label_list.append(index + 1)
            else:
                if 0.66 < ((battery.get_cycle(index)['Q discharge/mA.h'].max())/NCM_NCA_total_capacity < 1.05
                        and not battery.get_CCCV_stage(index).empty):
                    one_bat_dict['summary'].append((battery.get_cycle(index)['Q discharge/mA.h'].max())/NCM_NCA_total_capacity)
                else:
                    wrong_label_list.append(index)

        one_bat_dict['cycle'] = {}
        temp_bat_dict['cycle'] = {}
        for cyc in range(1, battery.cycle_life + 1):
            if int(battery.cycle_index.min()) > 1:
                if cyc + 1 not in wrong_label_list and not battery.get_CCCV_stage(cyc + 1).empty:
                    temp_bat_dict['cycle'][cyc - 1] = {}
                    for key in key_name:
                        temp_bat_dict['cycle'][cyc - 1][key] = battery.get_CCCV_stage(cyc + 1)[key][1:]

            else:
                if cyc not in wrong_label_list and not battery.get_CCCV_stage(cyc).empty:
                    temp_bat_dict['cycle'][cyc - 1] = {}
                    for key in key_name:
                        # 处理一些下标错误的cycle_index
                        temp_bat_dict['cycle'][cyc - 1][key] = battery.get_CCCV_stage(cyc)[key][1:]


        index = 0
        for idx in temp_bat_dict['cycle'].keys():
            one_bat_dict['cycle'][index] = {}
            for key in key_name:
                if temp_bat_dict['cycle'][idx][key].empty:
                    print("This Series is empty")
                one_bat_dict['cycle'][index][key] = temp_bat_dict['cycle'][idx][key]
            index += 1
            one_battery_len.append(temp_bat_dict['cycle'][idx]['<I>/mA'].shape[0])

        all_bat_dict[bat_idx] = one_bat_dict
        bat_idx += 1

        battery_average_len.append(sum(one_battery_len) / len(one_battery_len))

    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, 'Tongji_NCM_NCA_prepocess.pkl')
    save_all_battery2pkl(filepath, all_bat_dict)


