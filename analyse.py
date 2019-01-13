import pandas as pd
import matplotlib.pyplot as plt

def read_log(name):
    result = []
    with open('log/'+name,'r') as f:
        for line in f:
            line = line.split(' ')[1:]
            t = line[0].split(',')[0]
            round = line[0].split('-')[-1][:-1]
            accuracy = line[1][:-1]
            result.append([t,round,accuracy])
    result = result[:-1]
    df = pd.DataFrame(result,columns=['time','round','accuracy'])
    return df
if __name__ == '__main__':
    df = read_log('baseline_relu_64_1_100.log')
    print(df.columns)
    df.plot(x = df.time, y = df.accuracy)

