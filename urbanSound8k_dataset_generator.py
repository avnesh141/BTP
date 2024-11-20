#!/usr/bin/env python
# coding: utf-8

# ***Import Libraries And Noise Types Defined***

# In[63]:


import numpy as np
import os
import random
from pydub import AudioSegment


noise_class_dictionary = {
0 : "air_conditioner",
1 : "car_horn",
2 : "children_playing",
3 : "dog_bark",
4 : "drilling",
5 : "engine_idling",
6 : "gun_shot",
7 : "jackhammer",
8 : "siren",
9 : "street_music"
}

noise_class_dictionary


# In[64]:


dataset="Datasets/"
urbanData="UrbanSound8K/"

fold_names = []
for i in range(1,11):
    fold_names.append("fold"+str(i)+"/")

fold_names


# In[65]:


Urban8Kdir =urbanData
target_folder = dataset+"clean_trainset_28spk_wav"


# ***Get Files With Different Noise Type***

# In[66]:


def diffNoiseType(files,noise_type):
    result = []
    for i in files:
        if i.endswith(".wav"):
            fname = i.split("-")
            if fname[1] != str(noise_type):
                result.append(i)
    return result


# ***Files With Same Noise Type***

# In[67]:


def oneNoiseType(files, noise_type):
    result = []
    for i in files:
        if i.endswith(".wav"):
            fname = i.split("-")
            if fname[1] == str(noise_type):
                result.append(i)
    return result


# ***Generate Noisy File From Clean File***

# In[68]:


# def genNoise(filename, num_per_fold, dest):
#     clean_audio_path = target_folder+"/"+filename
#     audio_1 = AudioSegment.from_file(clean_audio_path)
#     counter = 0
#     for fold in fold_names:
#         dirname = Urban8Kdir + fold
#         # print(dirname)
#         dirlist = os.listdir(dirname)
#         total_noise_len = len(dirlist)
#         samples = np.random.choice(total_noise_len, num_per_fold, replace=False)
#         print(samples)
#         for s in samples:
#             noisefile = dirlist[s]
#             try:
#                 audio_2 = AudioSegment.from_file(dirname+"/"+noisefile)
#                 combined = audio_1.overlay(audio_2, times=5)
#                 target_dest = dest+"/"+filename[:len(filename)-4]+"_noise_"+str(counter)+".wav"
#                 combined.export(target_dest, format="wav")
#                 audio = Audio(target_dest)
#                 display(audio)
#                 counter +=1
#             except:
#                 print("Error")



# In[69]:


# genNoise("p226_001.wav",1,"Datasets")


# In[70]:


import torchaudio
from IPython.display import Audio


# In[71]:


def genNoisyFile(filename,dest, noise_type,isdiff):
    succ = False
    true_path = target_folder+"/"+filename
    count=0
    while not succ:
        try:
            audio_1 = AudioSegment.from_file(true_path)
        except:
            print("Some kind of audio decoding error occurred for base file... skipping")
            break
        try:
            fold =  np.random.choice(fold_names, 1, replace=False)
            fold = fold[0]
            dirname = Urban8Kdir + fold
            dirlist = os.listdir(dirname)
            noisefile=""
            if isdiff:
                possible_noises=diffNoiseType(dirlist,noise_type)
                total_noise = len(possible_noises)
                samples = np.random.choice(total_noise, 1, replace=False)
                s = samples[0]
                noisefile = possible_noises[s] 
            else:
               possible_noises=oneNoiseType(dirlist,noise_type)
               total_noise = len(possible_noises)
               samples = np.random.choice(total_noise, 1, replace=False)
               s = samples[0]
               noisefile = possible_noises[s]

            audio_2 = AudioSegment.from_file(dirname+"/"+noisefile)
            combined = audio_1.overlay(audio_2, times=5)
            target_dest = dest+"/"+filename
            combined.export(target_dest, format="wav")
            # audio =Audio(target_dest)
            # display(audio)
            succ = True
        except:
            if count>5:
                succ=True
            count+=1
            pass
            # print("Some kind of audio decoding error occurred for the noise file..retrying")
            # break;



# In[72]:


noise_type = int(input("Enter the noise class dataset to generate :\t"))


# In[73]:


inp_folder = dataset+"US_Class2"+str(noise_type)+"_Train_Input"
op_folder = dataset+"US_Class2"+str(noise_type)+"_Train_Output"


# In[74]:


print("Generating Training Data..")
print("Making train input folder")
if not os.path.exists(inp_folder):
    os.makedirs(inp_folder)
print("Making train output folder")
if not os.path.exists(op_folder):
    os.makedirs(op_folder)


# In[75]:


from tqdm import tqdm
print(os.listdir(target_folder).__len__())


# In[76]:


def generate_Train_dataset():
    counter = 0
    #noise_type = 1
    for file in tqdm(os.listdir(target_folder)):
        filename = os.fsdecode(file)
        if filename.endswith(".wav"):
            snr = random.randint(0,10)
            # noise_type=random.randint(0,9)
            genNoisyFile(filename,inp_folder,noise_type,0)
            snr = random.randint(0,10)
            genNoisyFile(filename,op_folder,noise_type,1)
            counter +=1
        if counter>=1000:
            break



# In[77]:


generate_Train_dataset()


# In[78]:


Urban8Kdir = urbanData
target_folder =  dataset+"clean_testset_wav"
inp_folder = dataset+"US_Class2"+str(noise_type)+"_Test_Input"
print(os.listdir(target_folder).__len__())


# In[79]:


print("Generating Testing Data..")
print("Making test input folder")
if not os.path.exists(inp_folder):
    os.makedirs(inp_folder)


# In[80]:


def generate_Test_dataset():
    counter = 0
    for file in tqdm(os.listdir(target_folder)):
        filename = os.fsdecode(file)
        if filename.endswith(".wav"):
            snr = random.randint(0,10)
            genNoisyFile(filename,inp_folder,noise_type,0)
            counter +=1
        if counter>=100:
            break


# In[81]:


generate_Test_dataset()


# In[ ]:





# In[ ]:




