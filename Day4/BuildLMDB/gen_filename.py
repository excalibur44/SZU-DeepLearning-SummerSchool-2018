import os

path = 'training'
with open("buildLMDB/" + path + ".txt", 'a+') as f:
    for d in os.listdir(path):
        for filename in os.listdir(path + '/' + d):
            s = d + '/' + filename + " " + d + '\n'
            f.write(s)
            #print(s)

path = 'testing'
with open("buildLMDB/" + path + ".txt", 'a+') as f:
    for d in os.listdir(path):
        for filename in os.listdir(path + '/' + d):
            s = d + '/' + filename + " " + d + '\n'
            f.write(s)
            #print(s)
