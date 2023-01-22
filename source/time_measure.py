#Script to measure execution times
import subprocess
import numpy as np
#sizes = [1024,2048,4096,8192,16384]
sizes = [32768]

Nit = 10
Nsizes = len(sizes)

#Minimum FFTW Estimate run time-------------------------------------------
execTimes = np.zeros(Nsizes)
execTimes += 1e10

execTimesPlan = np.zeros(Nsizes)
execTimesPlan += 1e10

for N in range(Nsizes):
    for it in range(Nit):
        args = ("../source/fftwCode", str(sizes[N]), "1")
        popen = subprocess.Popen(args, stdout=subprocess.PIPE)
        popen.wait()
        output = str(popen.stdout.read())
        output = output.split(" ",1)
        execTimes[N]= min(execTimes[N],float(output[1][0:-1]))
        execTimesPlan[N] = min(execTimesPlan[N], float(output[0][2:-1]))

f = open("../homog_times/fftwEstimate.txt", "w")
fp = open("../homog_times/fftwEstimatePlan.txt", "w")
for N in range(Nsizes):
    print("Size ", sizes[N], " time:", execTimes[N])
    f.write("Size "+str(sizes[N])+" time: "+str(execTimes[N])+"s\n")
    fp.write("Size "+str(sizes[N])+" time: "+str(execTimesPlan[N])+"s\n")

#Minimum FFTW Measure run time-------------------------------------------
execTimes = np.zeros(Nsizes)
execTimes += 1e10

execTimesPlan = np.zeros(Nsizes)
execTimesPlan += 1e10

for N in range(Nsizes):
    for it in range(Nit):
        args = ("../source/fftwCode", str(sizes[N]), "2")
        popen = subprocess.Popen(args, stdout=subprocess.PIPE)
        popen.wait()
        output = str(popen.stdout.read())
        output = output.split(" ",1)
        execTimes[N]= min(execTimes[N],float(output[1][0:-1]))
        execTimesPlan[N] = min(execTimesPlan[N], float(output[0][2:-1]))

f = open("../homog_times/fftwMeasure.txt", "w")
fp = open("../homog_times/fftwMeasurePlan.txt", "w")
for N in range(Nsizes):
    print("Size ", sizes[N], " time:", execTimes[N])
    f.write("Size "+str(sizes[N])+" time: "+str(execTimes[N])+"s\n")
    fp.write("Size "+str(sizes[N])+" time: "+str(execTimesPlan[N])+"s\n")

#Our minimum run time-------------------------------------------
execTimes = np.zeros(Nsizes)
execTimes += 1e10
for N in range(Nsizes):
    for it in range(Nit):
        args = ("../source/deriche_block_filter", str(sizes[N]))
        popen = subprocess.Popen(args, stdout=subprocess.PIPE)
        popen.wait()
        output = popen.stdout.read()
        execTimes[N]= min(execTimes[N],float(output))
f = open("../homog_times/ours.txt", "w")
for N in range(Nsizes):
    print("Size ", sizes[N], " time:", execTimes[N])
    f.write("Size "+str(sizes[N])+" time: "+str(execTimes[N])+"s\n")

