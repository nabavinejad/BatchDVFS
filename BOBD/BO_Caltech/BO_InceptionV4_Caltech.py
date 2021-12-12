import math
import numpy as np
import os
import time
import csv


############################# MUST SET  ##############
powerConstraint = 245
minSamples = 50
DNN_Name = "InceptionV4"
Input_Image_Folder = 'Caltech_256_20000'
DVFS_Levels = [544, 632, 734, 835, 949, 1063, 1189, 1303, 1430, 1531]

############################# END MUST SET  ##############
def evaluate(job_id, params):

    time.sleep(3)
    DVFS_Index = params['DVFS_Index']
    DVFS = DVFS_Levels[int(DVFS_Index[0])]
    BS = params['BS']

    os.system("echo password_of_user | sudo -S nvidia-smi --applications-clocks=3615," + str(DVFS))  # Set DVFS

    cmd = "python BatchDVFS_GPU_{3}.py " \
              " --native  --topN 1 --batch_size {1}   --image_folder {4}" \
              "  --nvidiasmi_outputFile '{2}_{3}_{4}_DVFS_{0}_BS_{1}_power.csv' " \
          " --latencyFile '{2}_{3}_{4}_DVFS_{0}_BS_{1}_latency.txt' --stop_time 60".format(str(DVFS), BS[0], job_id, DNN_Name, Input_Image_Folder)
    tic = time.time()
    os.system(cmd)
    toc = time.time()

    os.system("echo password_of_user | sudo -S nvidia-smi --reset-applications-clocks")

    time.sleep(3)


    latencyFile = "{2}_{3}_{4}_DVFS_{0}_BS_{1}_latency.txt".format(str(DVFS), BS[0], job_id, DNN_Name, Input_Image_Folder)
    firsline = 0
    latency = []
    for line in open(latencyFile):
        firsline = firsline + 1
        if firsline == 1:
            continue
        nums = line.split()  # split the line into a list of strings by whitespace
        index = 0
        nums2 = []
        for j in nums:
            nums2.append(float(j))  # turn each string into a float

        latency.append(nums2[0])

    #latency_95th = 1000 * np.percentile(latency, 95)  # 95 percentile latency in milisecond (ms)
    latency_average = np.mean(latency)
    throughput = (1/latency_average) * int(BS[0]) * -1   # -1 as it minimizes, but we want maximization


    fileName = "{2}_{3}_{4}_DVFS_{0}_BS_{1}_power.csv".format(str(DVFS), BS[0], job_id, DNN_Name, Input_Image_Folder)

    power = []

    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1  # skip the first line of the filehich is the name of columns
                continue
            else:
                line_count += 1
                power.append(float(row[0]))
    maxPower = np.max(power)
    power_con = float(powerConstraint - maxPower)  # power <= constraint



    # if job_id <= minSamples:
    #     powerResult = open('MaxPower_Job.txt','w')
    #     powerResult.write(str(maxPower))
    #     powerResult.close()
    # if job_id > minSamples:
    #     powerResult = open('MaxPower_Job.txt', 'r')
    #     test_lines = powerResult.readline()
    #     powerResult.close()
    #     previousPower = float(test_lines[0])
    #     if (abs(previousPower - maxPower))/previousPower < 0.05:
    #         if os.path.exists('Stopping_Conditino.txt'):
    #             append_write = 'a'  # append if already exists
    #         else:
    #             append_write = 'w'  # make a new file if not
    #
    #         highscore = open('Stopping_Conditino.txt', append_write)
    #         highscore.write(job_id  + ',' + previousPower + ',' + maxPower + '\n')
    #         highscore.close()
    #     else:
    #         powerResult = open('MaxPower_Job.txt', 'w')
    #         powerResult.write(str(maxPower))
    #         powerResult.close()

    if os.path.exists("BatchDVFS_{0}_{1}_Log.txt".format(DNN_Name,Input_Image_Folder)):
        append_write = 'a'  # append if already exists

    else:
        append_write = 'w'  # make a new file if not
    totalResult = open("BatchDVFS_{0}_{1}_Log.txt".format(DNN_Name,Input_Image_Folder), append_write)
    if append_write == 'w':
        totalResult.write("job_id, DVFS, BS, Average_Latency(s), Power_Const (W), Power (W), Throughput, Runtime(s)\n")
    totalResult.write(
        str(job_id) + "," + str(DVFS) + "," + str(BS[0]) + "," + str(latency_average) + "," + str(powerConstraint)
        + "," + str(maxPower) + "," + str(throughput * (-1)) + "," + str(toc - tic))
    totalResult.write("\n")
    totalResult.close()

    if int(job_id) == minSamples:
        processID = open('processID.txt', 'r+')
        lines = processID.readlines()
        #processID.truncate(0)
        processID.close()
        cmd = "kill {0}".format(lines[0])
        os.system(cmd)


    return {
        "constraint_power": power_con,
        "objective_throughput": throughput
    }

    # True minimum is at 2.945, 2.945, with a value of 0.8447


def main(job_id, params):

    try:
        return evaluate(job_id, params)
    except Exception as ex:
        print(ex)
        print('An error occurred in branin_con.py')
        return np.nan