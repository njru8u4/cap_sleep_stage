import mne
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import convolve, find_peaks
import time
#from numba import jit

def stage_trans(stage_name):
    if stage_name == 'W' or stage_name == 'Wake' or stage_name == 'w' or stage_name == 'w':
        return 0
    elif stage_name == 'S1' or stage_name == 's1':
        return 1
    elif stage_name == 'S2' or stage_name == 's2':
        return 2
    elif stage_name == 'S3' or stage_name == 's3':
        return 3
    elif stage_name == 'S4' or stage_name == 's4':
        return 4
    elif stage_name == 'REM' or stage_name == 'R':
        return 5
    else:
        write_error(f"{stage_name}\t unknown stage name\n")
        return 6
   
def write_error(massage):
    f = open('./datas/errors.txt', 'a')
    f.write(massage)
    f.close()


def read_PPG_from_edf(file_path, time, stage):
    start_time = time[0]
    for i in range(len(time)):
        if time[i] > start_time:
            time[i] = time[i] - start_time
        else:
            timeStamp = 24*60*60
            time[i] = timeStamp - start_time + time[i]


    data = mne.io.read_raw_edf(file_path)
    sample_f = int(data.info['sfreq'])

    #raw, times = data.get_data(picks='ECG1-ECG2', return_times=True)
    try:
        raw = data.get_data(picks='PLETH')
    except:
        try:
            raw = data.get_data(picks='Pleth')
        except:
            write_error(f"{file_path}\t pleth load error\n")
            return 1, 1
            #exit()

    raw = raw.reshape(len(raw[0]))
    raw = -raw
    mid_filter_num = int(sample_f / 25 / 2) + 1
    filt = np.array([1/mid_filter_num]*mid_filter_num)
    
    raw = convolve(raw, filt)

    temp_peak = []
    temp_stage = []
    for i in range(len(time)):
        interval = np.zeros(100)
        peaks, _ = find_peaks(raw[time[i]*sample_f:time[i]*sample_f+sample_f*30], distance= int(sample_f / 3), width= int(sample_f / 6))
        
        if peaks.size < 15 or peaks.size > 100:
            write_error(f"{file_path}\t have peak error at time stamp: {time[i]}\n")
            continue
        interval[:len(peaks)] = peaks/sample_f*1000
        temp_stage.append(stage_trans(stage[i]))
        temp_peak.append(interval)
    #temp_peak = np.array(temp_peak)
    #print(temp_peak.shape)
    return temp_peak, temp_stage

             

def read_stage(file_path, interval_time):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    start_index = 9999
    style = 0
    temp_stage = []
    temp_time = []
    for i in range(len(lines)):
        if i < start_index:
            if lines[i] == "Sleep Stage	Time [hh:mm:ss]	Event	Duration[s]	Location\n":
                start_index = i
                style = 1
            elif lines[i] == "Sleep Stage	Position	Time [hh:mm:ss]	Event	Duration[s]	Location\n":
                start_index = i
                style = 2
            else:
                continue
        elif i > start_index and i > (interval_time / 30 - 1) / 2 + start_index:
            temp = lines[i].split("\t")
            try:
                if style == 1:
                    if temp[3] != '30':
                        continue
                    try:
                        timeArray = time.strptime(temp[1], "%H:%M:%S")
                    except:
                        try:
                            timeArray = time.strptime(temp[1], "%H.%M.%S")
                        except:
                            write_error(f"{file_path}\t time error\n")
                            exit()
                    timeStamp = timeArray.tm_hour*3600 + timeArray.tm_min*60 + timeArray.tm_sec
                    temp_time.append(timeStamp)
                    temp_stage.append(temp[0])
                elif style == 2:
                    if temp[4] != '30':
                        continue
                    try:
                        timeArray = time.strptime(temp[2], "%H:%M:%S")
                    except:
                        try:
                            timeArray = time.strptime(temp[2], "%H.%M.%S")
                        except:
                            write_error(f"{file_path}\t time error\n")
                            break
                    timeStamp = timeArray.tm_hour*3600 + timeArray.tm_min*60 + timeArray.tm_sec
                    #timeStamp = int(time.mktime(timeArray))
                    temp_time.append(timeStamp)
                    temp_stage.append(temp[0])
                #print(temp)
            except:
                
                if i - start_index > 3:
                    write_error(f"{file_path}\t duration warning\n")
                    break
                else:
                    write_error(f"{file_path}\t duration error\n")
                    exit()
    #temp_stage = np.array(temp_stage)
    temp_time = np.array(temp_time)
    print (temp_time.shape)
    if temp_time.shape[0] == 0:
        write_error(f"{file_path}\t load txt error\n")
        return 1, 1
    #print (f"stage shape = {temp_stage.shape}")
    #print (f"time shape = {temp_time.shape}")
    return temp_stage, temp_time


def read_files(interval_time):
    if os.path.exists("./datas/errors.txt"):
        os.remove("./datas/errors.txt")
    record = open('./cap-sleep-database-1.0.0/RECORDS', 'r')
    lines = record.readlines()
    all_stages = []
    all_PPG = []
    
    for i in range(len(lines)):
        temp = lines[i].split('.')
        txt_file = './cap-sleep-database-1.0.0/' + temp[0] + '.txt'
        edf_file = './cap-sleep-database-1.0.0/' + temp[0] + '.edf'
        """
        try:
            stage, time = read_stage(txt_file)
            PPG, stage = read_PPG_from_edf(edf_file, time, stage)
        except:
            failed_files.append(temp[0])
            print(f"{temp[0]}\t open failed!!!")
            continue
        """
        stage, time = read_stage(txt_file, interval_time)
        if stage == 1 and time == 1:
            continue
        #print ("1")
        PPG, stage = read_PPG_from_edf(edf_file, time, stage)
        if stage == 1 and PPG == 1:
            continue

        all_stages = all_stages + stage
        all_PPG = all_PPG + PPG

    all_PPG = np.array(all_PPG)
    all_stages = np.array(all_stages)
    
    print (all_PPG.shape)
    print (all_stages.shape)
    stage_count = [0]*7
    for i in range(len(all_stages)):
        stage_count[all_stages[i]] += 1
    print (stage_count)
    np.save('./datas/cap_PPG_30s_initial', all_PPG)
    np.save('./datas/cap_label_30s_initial', all_stages)
    #print (failed_files)

    return all_PPG, all_stages

def read_test(name):
    data = mne.io.read_raw_edf(name)
    print (data.info)
    print (data.info['sfreq'])
    print (data.info['ch_names'])

if __name__ == "__main__":
    #read_test("./cap-sleep-database-1.0.0/sdb3.edf")
    """
    stage, time = read_stage("./cap-sleep-database-1.0.0/n14.txt")
    if stage == 1 and time == 1:
        print ("fail")
    PPG, stage = read_PPG_from_edf("./cap-sleep-database-1.0.0/n14.edf", time, stage)
    """
    interval_time = 30
    PPG, stage = read_files(interval_time)