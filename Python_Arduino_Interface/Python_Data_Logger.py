'''
This script is inspired by:
https://engineersportal.com/blog/2018/2/25/python-datalogger-reading-the-serial-output-from-arduino-to-analyze-data-using-pyserial 
'''

import serial
import time
import csv
import matplotlib
matplotlib.use("tkAgg")
import matplotlib.pyplot as plt
import numpy as np
import datetime
import argparse

ser = serial.Serial('/COM3')
ser.flushInput()

plot_window = 500
y_var = np.array(np.zeros([plot_window]))

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(y_var)

#region: parsing arguements passed via CMD line
parser = argparse.ArgumentParser(
    description='Test code to learn how scripts with args work',
    epilog="For more information call me baby")

parser.add_argument('-n', '--name', type=str, default="test_result", help='name of the test to save in csv format')
args = vars(parser.parse_args())
output_csv_name = args['name']
while True:
    try:
        ser_bytes = ser.readline()
        try:
            decoded_bytes = float(ser_bytes[0:len(ser_bytes)-2].decode("utf-8"))
            print(decoded_bytes)
        except:
            continue
        with open(output_csv_name+ ".csv","a", newline='') as f:
            writer = csv.writer(f,delimiter=",")
            currentDT = datetime.datetime.now()
            writer.writerow([currentDT.strftime("%H:%M:%S:%f")[:-4],decoded_bytes])
        y_var = np.append(y_var,decoded_bytes)
        y_var = y_var[1:plot_window+1]
        line.set_ydata(y_var)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()
    except:
        print("Keyboard Interrupt")
        break