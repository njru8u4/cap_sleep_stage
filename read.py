import mne
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import convolve, find_peaks
import time
import shutil

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
   
def write_error(message):
    f = open('./datas/errors.txt', 'a')
    f.write(message)
    f.close()

def write_info(message):
    f = open('./file_info.txt', 'a')
    f.write(message)
    f.close()

def save_pic(sig, timeStamp, file_name, size):
    temp = file_name.split('/')
    plt.plot(sig)
    plt.title(f"peaks = {size}")
    plt.savefig(f"./pictures/{temp[-1]}_at_{timeStamp}.png")
    plt.close()

def get_timeStamp(str):
    try:
        timeArray = time.strptime(str, "%H:%M:%S")
    except:
        timeArray = time.strptime(str, "%H.%M.%S")
    return timeArray.tm_hour*3600 + timeArray.tm_min*60 + timeArray.tm_sec
    

def read_PPG_from_edf(file_path, times, stage, interval_time):
    data = mne.io.read_raw_edf(file_path)
    if data.info['sfreq'].is_integer():
        sample_f = int(data.info['sfreq'])
    else:
        write_error(f"{file_path}\t sample rate not integer\n")
        return 1, 1

    #raw, times = data.get_data(picks='ECG1-ECG2', return_times=True)
    try:
        raw = data.get_data(picks='PLETH')
    except:
        try:
            raw = data.get_data(picks='Pleth')
        except:
            write_error(f"{file_path}\t pleth load error\n")
            return 1, 1

    #### dealing time ##########
    #timeArray = time.strptime(data.info['meas_date'], "%Y-%m-%d %H:%M:%S+00:00")
    #start_time = timeArray.tm_hour*3600 + timeArray.tm_min*60 + timeArray.tm_sec
    start_time = data.info['meas_date'].hour * 3600 + data.info['meas_date'].minute * 60 + data.info['meas_date'].second
    for i in range(len(times)):
        if times[i] >= start_time:
            times[i] = times[i] - start_time
        else:
            times[i] = 24*3600 - start_time + times[i]


    raw = raw.reshape(len(raw[0]))
    raw = -raw
    mid_filter_num = int(sample_f / 25 / 2) + 1
    filt = np.array([1/mid_filter_num]*mid_filter_num)
    raw = convolve(raw, filt)

    temp_peak = []
    temp_stage = []
    for i in range(len(times)):
        interval = np.zeros(100*int(interval_time / 30))
        start = int(times[i] - (interval_time - 30) / 2) * sample_f
        end = int(times[i] + 30 + (interval_time - 30) / 2) * sample_f
        if end > len(raw):
            write_error(f"{file_path}\t time overflow, rest epoch {len(times) - i}\n")
            break
        peaks, _ = find_peaks(raw[start : end], distance= int(sample_f / 3), width= int(sample_f / 8))
        
        if peaks.size < 15*interval_time/30 or peaks.size > 100*interval_time/30:
            write_error(f"{file_path}\t have peak error at time stamp: {times[i]}\n")
            save_pic(raw[start: end], times[i], file_path, peaks.size)
            continue
        peak2sec = peaks/sample_f*1000
        real_peak = peak2sec[1:] - peak2sec[0:-1]
        interval[:len(real_peak)] = real_peak
        temp_stage.append(stage_trans(stage[i]))
        temp_peak.append(interval)
    
    return temp_peak, temp_stage

def read_stage(file_path, interval_time):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    start_index = 9999
    style = 0
    temp_stage = []
    temp_time = []

    start_time = 0
    for i in range(len(lines)):
        if i < start_index:
            if lines[i] == "Sleep Stage	Time [hh:mm:ss]	Event	Duration[s]	Location\n":
                start_index = i
                style = 1
            elif lines[i] == "Sleep Stage	Position	Time [hh:mm:ss]	Event	Duration[s]	Location\n":
                start_index = i
                style = 2
            elif lines[i] == "Sleep Stage	Position	Time [hh:mm:ss]	Event	Duration [s]	Location\n":
                start_index = i
                style = 2
                print ("in")
            else:
                continue

        if i == start_index + 1:
            temp = lines[i].split("\t")
            start_time = get_timeStamp(temp[style])
            
        if i > (interval_time / 30 - 1) / 2 + start_index and i + (interval_time / 30 - 1) / 2 < len(lines):
            if style == 0:
                write_error(f"{file_path}\t time column error\n")
                return 1, 1
            
            temp = lines[i].split("\t")
            try:
                if temp[style+2] != '30':
                    continue
                timeStamp = get_timeStamp(temp[style])
                temp_time.append(timeStamp)
                temp_stage.append(temp[0])
            except:
                if i - start_index > 3:
                    write_error(f"{file_path}\t duration warning\n")
                    break
                else:
                    write_error(f"{file_path}\t duration error\n")
                    exit()
    times = np.array(temp_time)
    if times.shape[0] == 0:
        write_error(f"{file_path}\t load txt error\n")
        return 1, 1

    

    return temp_stage, times

def read_files(interval_time):
    if os.path.exists("./datas/errors.txt"):
        os.remove("./datas/errors.txt")
    if os.path.exists('./file_info.txt'):
        os.remove('./file_info.txt')
    if os.path.exists("./pictures/"):
        shutil.rmtree('./pictures')
        time.sleep(1)
        
    os.mkdir('./pictures/')
    record = open('./cap-sleep-database-1.0.0/RECORDS', 'r')
    lines = record.readlines()
    all_stages = []
    all_PPG = []
    
    for i in range(len(lines)):
        temp = lines[i].split('.')
        txt_file = './cap-sleep-database-1.0.0/' + temp[0] + '.txt'
        edf_file = './cap-sleep-database-1.0.0/' + temp[0] + '.edf'
        
        stage, times = read_stage(txt_file, interval_time)
        if stage == 1 and times == 1:
            continue
        PPG, stage = read_PPG_from_edf(edf_file, times, stage, interval_time)
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
    return all_PPG, all_stages

def read_test(name):
    data = mne.io.read_raw_edf(name)
    sample_f = int (data.info['sfreq'])
    raw = data.get_data(picks='PLETH')
    raw = raw.reshape(len(raw[0]))
    
    #plt.plot(raw[600*sample_f:610*sample_f])
    #plt.show()
    
    
    
    raw = -raw
    #plt.plot(raw[600*sample_f:610*sample_f])
    #plt.show()
    

    mid_filter_num = int(sample_f / 25 / 2) + 1
    filt = np.array([1/mid_filter_num]*mid_filter_num)
    raw = convolve(raw, filt)
    plt.plot(raw[60*sample_f:70*sample_f])
    plt.show()
    


    """
    print (data.info)
    print (data.info['sfreq'])
    print (data.info['meas_date'])
    print (data.info['meas_date'].second)
    print (data.info['meas_date'].minute)
    """


if __name__ == "__main__":
    read_test("./cap-sleep-database-1.0.0/plm10.edf")
    
    #stage, time = read_stage("./cap-sleep-database-1.0.0/n16.txt", 150)
    #if stage == 1 and time == 1:
    #    print ("fail")
    #PPG, stage = read_PPG_from_edf("./cap-sleep-database-1.0.0/n14.edf", time, stage)
    
    #interval_time = 30
    #PPG, stage = read_files(interval_time)
    #np.save('./datas/cap_PPG_30s', PPG)
    #np.save('./datas/cap_label_for_PPG_30s', stage)
    