'''
Author: Fujin Wang
Date: 2024.05
Github: https://github.com/wang-fujin

Description:
该代码用于读取MIT电池数据集，方便后续预处理和分析。
'''
import pickle

import pandas as pd
import numpy as np
import h5py
import pprint
import matplotlib.pyplot as plt
import functools
from tqdm import tqdm

from scipy import interpolate
from scipy.signal import resample



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
                    try:
                        f1 = interpolate.interp1d(x, data[k], kind='linear')
                        new_df[k] = f1(new_x)
                    except ValueError:
                        print(data.shape[0])
            return new_df
        return wrapper
    return decorator


class BatchBattery:
    def __init__(self, path):
        self.path = path
        self.f = h5py.File(path, 'r')
        self.batch = self.f['batch']
        self.date = self.f['batch_date'][:].tobytes()[::2].decode()
        self.num_cells = self.batch['summary'].shape[0]
        self.exclude = {}
        for idx in range(self.num_cells):
            self.exclude[idx] = []

        print('date: ', self.date)
        print('num_cells: ', self.num_cells)

    def get_cell_nums(self):
        return self.num_cells

    def get_one_battery_cycle_num(self, cell_num):
        _, bat_cycle_dict = self.get_one_battery(cell_num)
        return len(bat_cycle_dict)

    def get_one_battery_one_cycle_Qd(self, cell_num, cycle_num):
        summary, _ = self.get_one_battery(cell_num)
        return summary['QD'][cycle_num]

    def get_one_battery(self, cell_num):
        '''
        读取一个电池的数据
        :param cell_num: 电池序号
        :return:
        '''
        i = cell_num
        f = self.f
        batch = self.batch
        cl = self.f[batch['cycle_life'][i, 0]][:][0][0]
        policy = self.f[batch['policy_readable'][i, 0]][:].tobytes()[::2].decode()
        # barcode = self.f[batch['barcode'][i, 0]].value.tobytes()[::2].decode()
        # channel_ID = self.f[batch['channel_id'][i, 0]].value.tobytes()[::2].decode()
        # summary_IR = np.hstack(self.f[batch['summary'][i, 0]]['IR'][0, :].tolist())
        # summary_QC = np.hstack(self.f[batch['summary'][i, 0]]['QCharge'][0, :].tolist())
        # summary_TA = np.hstack(self.f[batch['summary'][i, 0]]['Tavg'][0, :].tolist())
        # summary_TM = np.hstack(self.f[batch['summary'][i, 0]]['Tmin'][0, :].tolist())
        # summary_TX = np.hstack(self.f[batch['summary'][i, 0]]['Tmax'][0, :].tolist())
        # summary_CT = np.hstack(self.f[batch['summary'][i, 0]]['chargetime'][0, :].tolist())
        summary_QD = np.hstack(self.f[batch['summary'][i, 0]]['QDischarge'][0, 1:].tolist())
        summary_CY = np.hstack(self.f[batch['summary'][i, 0]]['cycle'][0, 1:].tolist())
        true_cl = len(summary_QD)
        summary = {'QD': summary_QD, 'cycle_life':true_cl, 'policy': policy,
                   'cycle_index': summary_CY.astype(int)}
        cycles = f[batch['cycles'][i, 0]]

        # 解析cycle的数据
        cycle_dict = {}
        for j in range(1,cycles['I'].shape[0]):
            I = np.hstack((f[cycles['I'][j, 0]][:]))
            V = np.hstack((f[cycles['V'][j, 0]][:]))
            Qc = np.hstack((f[cycles['Qc'][j, 0]][:]))
            Qd = np.hstack((f[cycles['Qd'][j, 0]][:]))
            t = np.hstack((f[cycles['t'][j, 0]][:]))
            one_cycle = {'current (A)': I, 'voltage (V)': V, 'charge Q (Ah)': Qc,
                         'discharge Q (Ah)': Qd, 'time (min)': t}
            cycle_dict[j] = one_cycle

        return summary, cycle_dict

    def get_one_battery_one_cycle(self,cell_num, cycle_num):
        '''
        读取一个电池的某个cycle的数据
        :param cell_num: 电池序号
        :param cycle_num: cycle序号
        :return: DataFrame
        '''
        i = cell_num
        f = self.f
        batch = self.batch

        cycles = f[batch['cycles'][i, 0]]
        # 检查cycle_num是否合法
        if cycle_num >= cycles['I'].shape[0] or cycle_num < 1:
            raise ValueError('cycle_num must be in [1,{}]'.format(cycles['I'].shape[0]))

        # 解析cycle的数据
        j = cycle_num
        I = np.hstack((f[cycles['I'][j, 0]][:]))
        V = np.hstack((f[cycles['V'][j, 0]][:]))
        Qc = np.hstack((f[cycles['Qc'][j, 0]][:]))
        Qd = np.hstack((f[cycles['Qd'][j, 0]][:]))
        t = np.hstack((f[cycles['t'][j, 0]][:]))
        T = np.hstack((f[cycles['T'][j, 0]][:]))
        one_cycle = {'current (A)': I, 'voltage (V)': V, 'charge Q (Ah)': Qc,
                     'discharge Q (Ah)': Qd, 'time (min)': t, 'Temperature': T}
        cycle_df = pd.DataFrame(one_cycle)
        return cycle_df

    def get_one_battery_one_cycle_charge(self,cell_num,cycle_num):
        '''
        读取某个电池的某个cycle的充电阶段的数据
        :param cell_num: int: 电池序号
        :param cycle_num: int: cycle序号
        :return: DataFrame
        '''
        cycle_df = self.get_one_battery_one_cycle(cell_num, cycle_num)
        cycle_df = cycle_df[cycle_df['current (A)'] > -2e-1]
        index = cycle_df.index
        # 判断index是否连续。如果index分段连续，则取第一段index
        diff = np.diff(index)
        continuous_index = np.where(diff != 1)[0]

        if len(continuous_index) > 0:
            cycle_df = cycle_df.loc[:continuous_index[0]]
        else:
            cycle_df = cycle_df.loc[:]
        # cycle_df.plot(x='time (min)',y='current (A)',c='r',linewidth=2)
        # plt.show()
        # cycle_df.plot(x='time (min)',y='voltage (V)',c='r',linewidth=2)
        # plt.show()
        return cycle_df

    @interpolate_resample(resample=True, num_points=303)
    def get_one_battery_one_cycle_CCCV_stage(self, cell_num, cycle_num):
        '''
        读取某个电池的某个cycle的CCCV阶段的数据（SOC>80%的数据）
        :param cell_num: int: 电池序号
        :param cycle_num: int: cycle序号
        :return: DataFrame
        '''
        charge_df = self.get_one_battery_one_cycle_charge(cell_num, cycle_num)
        rough_CCCV_df = charge_df[charge_df['charge Q (Ah)'] >= 0.77 * charge_df['charge Q (Ah)'].max()]
        # rough_more_CCCV_df = charge_df[charge_df['charge Q (Ah)'] >= 0.70*charge_df['charge Q (Ah)'].max()]
        CCCV_df = rough_CCCV_df[rough_CCCV_df['current (A)'] > 0.01]
        # CCCV_more_df = rough_more_CCCV_df[rough_more_CCCV_df['current (A)'] > 0.001]
        diff = np.diff(CCCV_df.index)
        # charge_df.plot(x='time (min)',y='current (A)',c='r',linewidth=2)
        # plt.tight_layout()
        # plt.show()
        #
        # rough_CCCV_df.plot(x='time (min)', y='current (A)', c='r', linewidth=2)
        # plt.tight_layout()
        # plt.show()
        continuous_index = np.where(diff != 1)[0]

        if len(continuous_index) > 0:
            start_index = CCCV_df.index[continuous_index[0] + 1]
            CCCV_df = CCCV_df.loc[start_index:]

        if CCCV_df.shape[0] < 50:
            # print(CCCV_df)
            # if rough_CCCV_df.shape[0] < 50:
            print(f"{cell_num}: {cycle_num} using rough CCCV data!")
            # self.exclude[cell_num].append(cycle_num)
            return rough_CCCV_df
            # else:
            # print(f"do not cut CCCV_df at {cell_num}: {cycle_num}, pick rough one.")
            # CCCV_df.plot(x='time (min)', y='current (A)', c='b', linewidth=2)
            # plt.tight_layout()
            # plt.show()
            # print(charge_df)
            # charge_df.plot(x='time (min)', y='current (A)', c='b', linewidth=2)
            # plt.tight_layout()
            # plt.show()
            #     # print("CCCV_df len :", len(CCCV_df), "Rough_CCCV_df len :", len(rough_CCCV_df))
            #     CCCV_df = rough_CCCV_df
            #     return CCCV_df
        return CCCV_df
        # return CCCV_df

    def get_one_battery_one_cycle_CCCV_stage_raw(self, cell_num, cycle_num):
        '''
        读取某个电池的某个cycle的CCCV阶段的数据（SOC>80%的数据）
        :param cell_num: int: 电池序号
        :param cycle_num: int: cycle序号
        :return: DataFrame
        '''
        charge_df = self.get_one_battery_one_cycle_charge(cell_num, cycle_num)
        rough_CCCV_df = charge_df[charge_df['charge Q (Ah)'] >= 0.79*charge_df['charge Q (Ah)'].max()]
        CCCV_df = rough_CCCV_df[rough_CCCV_df['current (A)'] > 0.01]

        diff = np.diff(CCCV_df.index)

        continuous_index = np.where(diff != 1)[0]

        if len(continuous_index) > 0:
            start_index = CCCV_df.index[continuous_index[0] + 1]
            CCCV_df = CCCV_df.loc[start_index:]

        # CCCV_df.plot(x='time (min)',y='voltage (V)',c='b',linewidth=2)
        # plt.tight_layout()
        # plt.show()

        return CCCV_df


    def get_one_battery_one_cycle_CV_stage(self,cell_num,cycle_num,current_range=[0.5,0.1]):
        '''
        读取某个电池的某个cycle的CV阶段的数据（SOC>80%的数据）
        :param cell_num: int: 电池序号
        :param cycle_num: int: cycle序号
        :param current_range: tuple: 电流范围, 如 [0.5,0.1]
        :return: DataFrame
        '''
        CCCV_df = self.get_one_battery_one_cycle_CCCV_stage(cell_num,cycle_num)
        CV_df = CCCV_df[CCCV_df['voltage (V)'] > 3.595]
        if current_range is not None:
            CV_df = CV_df[(CV_df['current (A)'] > current_range[1]) & (CV_df['current (A)'] < current_range[0])]
        return CV_df


    def plot_one_battery_one_cycle_CCCV_stage(self,cell_num,cycle_num):
        '''
        在一个4行1列的图中画出一个电池的某个cycle的CCCV阶段的数据
        :param cell_num: int: 电池序号
        :param cycle_num: int: cycle序号
        :return: None
        '''
        CC_df = self.get_one_battery_one_cycle_CC_stage(cell_num,cycle_num)
        CV_df = self.get_one_battery_one_cycle_CV_stage(cell_num,cycle_num,current_range=None)

        fig,axes = plt.subplots(4,1,figsize=(6,6),dpi=200)
        axes[0].plot(CC_df['time (min)'],CC_df['current (A)'],c='r',linewidth=2)
        axes[0].plot(CV_df['time (min)'],CV_df['current (A)'],c='b',linewidth=2)
        axes[0].axvline(x=CC_df['time (min)'].values[-1],c='g',linewidth=1,linestyle='--')
        axes[0].set_ylabel('current (A)')

        axes[1].plot(CC_df['time (min)'],CC_df['voltage (V)'],c='r',linewidth=2)
        axes[1].plot(CV_df['time (min)'],CV_df['voltage (V)'],c='b',linewidth=2)
        axes[1].axvline(x=CC_df['time (min)'].values[-1], c='g', linewidth=1, linestyle='--')
        axes[1].set_ylabel('voltage (V)')

        axes[2].plot(CC_df['time (min)'],CC_df['charge Q (Ah)'],c='r',linewidth=2)
        axes[2].plot(CV_df['time (min)'],CV_df['charge Q (Ah)'],c='b',linewidth=2)
        axes[2].axvline(x=CC_df['time (min)'].values[-1], c='g', linewidth=1, linestyle='--')
        axes[2].set_ylabel('charge Q (Ah)')

        axes[3].plot(CC_df['time (min)'],CC_df['discharge Q (Ah)'],c='r',linewidth=2)
        axes[3].plot(CV_df['time (min)'],CV_df['discharge Q (Ah)'],c='b',linewidth=2)
        axes[3].axvline(x=CC_df['time (min)'].values[-1], c='g', linewidth=1, linestyle='--')
        axes[3].set_ylabel('discharge Q (Ah)')
        axes[3].set_xlabel('time (min)')

        plt.tight_layout()
        plt.show()



    def plot_one_battery_one_cycle(self,cell_num,cycle_num):
        '''
        在一个4行1列的图中画出一个电池的某个cycle的数据
        :param cell_num: int: 电池序号
        :param cycle_num: int: cycle序号
        :return:
        '''
        cycle_df = self.get_one_battery_one_cycle(cell_num,cycle_num)
        fig,axes = plt.subplots(4,1,figsize=(6,6),dpi=200)
        axes[0].plot(cycle_df['time (min)'],cycle_df['current (A)'],c='r',linewidth=2)
        axes[0].set_ylabel('current (A)')
        axes[1].plot(cycle_df['time (min)'],cycle_df['voltage (V)'],c='b',linewidth=2)
        axes[1].set_ylabel('voltage (V)')
        axes[2].plot(cycle_df['time (min)'],cycle_df['charge Q (Ah)'],c='g',linewidth=2)
        axes[2].set_ylabel('charge Q (Ah)')
        axes[3].plot(cycle_df['time (min)'],cycle_df['discharge Q (Ah)'],c='k',linewidth=2)
        axes[3].set_ylabel('discharge Q (Ah)')
        axes[3].set_xlabel('time (min)')
        plt.tight_layout()
        plt.show()

    def save_all_battery2pkl(self, all_battery):
        # Note that all_battery is a dictionary contains different Battery.
        with open('batch2_prepocess.pkl', 'wb') as fp:
            pickle.dump(all_battery, fp)
        print("file to pkl finished!")


if __name__ == '__main__':
    # 一个简单的例子
    path = '../cell_data/mit_dataset/2018-04-12_batchdata_updated_struct_errorcorrect.mat'  # Batch 3

    bb = BatchBattery(path)
    cell_nums = bb.get_cell_nums()
    all_battery = {}
    for bat_idx in tqdm(range(cell_nums), total=cell_nums):
        one_battery = {}
        one_battery['summary'] = []
        one_battery['cycle'] = {}
        summary, _ = bb.get_one_battery(bat_idx)
        # one_battery['cycle_raw'] = {}
        for cyc in range(1, int(bb.get_one_battery_cycle_num(bat_idx)) + 1):
            one_battery['cycle'][cyc-1] = bb.get_one_battery_one_cycle_CCCV_stage(bat_idx, cyc)
            one_battery['summary'].append(summary['QD'][cyc-1])
            # one_battery['cycle_raw'][cyc-1] = bb.get_one_battery_one_cycle_CCCV_stage_raw(bat_idx, cyc)
            # if one_battery['cycle_raw'][cyc-1].shape[0] < 10:
            #     fig, axes = plt.subplots(3, 1, figsize=(8, 6), dpi=200)
            #     axes[0].plot(one_battery['cycle_raw'][cyc - 1]['current (A)'])
            #     axes[0].set_ylabel('raw data')
            #     axes[1].plot(one_battery['cycle'][cyc - 1]['current (A)'])
            #     axes[1].set_ylabel('scipy interpolate')
            #     axes[2].plot(resample(one_battery['cycle_raw'][cyc-1]['current (A)'], 278))
            #     axes[2].set_ylabel('Fourier')
            #     fig.suptitle("Current")
            #     plt.tight_layout()
            #     plt.show()
        # for cyc in range(1, int(bb.get_one_battery_cycle_num(bat_idx)) + 1):
        #     if cyc not in bb.exclude[bat_idx]:
        #         one_battery['summary'].append(summary['QD'][cyc - 1])
        #
        # one_battery['summary'] = pd.DataFrame(one_battery['summary'])
        all_battery[bat_idx] = one_battery
        assert len(one_battery['cycle']) == len(one_battery['summary']), bat_idx
    # bb.save_all_battery2pkl(all_battery=all_battery)
    # average_len_over1bat = []
    # for bat_idx in range(cell_nums):
    #     sum = 0
    #     for cyc in range(1, len(all_battery[bat_idx]['cycle']) + 1):
    #         sum += all_battery[bat_idx]['cycle'][cyc-1]['current (A)'].shape[0]
    #     sum /= len(all_battery[bat_idx]['cycle'])
    #     average_len_over1bat.append(sum)
    # print(1)


    # cycle_num = 10
    # bb.plot_one_battery_one_cycle_CCCV_stage(all_battery[0], cycle_num)



