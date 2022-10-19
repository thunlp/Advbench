import argparse
from ast import arg
import dataset
import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from PackDataset import packDataset_util
from transformers import AutoTokenizer
from transformers import BertTokenizer, BertForSequenceClassification
from attack_util import *
from tqdm import tqdm
from nltk.corpus import stopwords
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import pandas as pd
import random
random.seed(714)
import os
# base_path = os.path.dirname(os.getcwd ()) 
base_path = os.path.abspath('.')
data_path = base_path +"/data/"

def load_data(data_name,type):
    file_path = data_path+data_name+"/"
    data = pd.read_csv(file_path+type+".csv")
    p_data = []
    for i in range(len(data)):
        p_data.append((data['text'][i], data['label'][i]))
    return p_data 


def get_output_label(sentence,tokenizer,model):
    if pd.isna(sentence):
        return -1
    tokenized_sent = tokenizer(sentence, return_tensors='pt', max_length=512, truncation=True)
    input_ids, attention_mask = tokenized_sent['input_ids'], tokenized_sent['attention_mask']
    # print(len(input_ids[0]))
    if torch.cuda.is_available():
        input_ids, attention_mask = input_ids.cuda(), attention_mask.cuda()
    output_label = model(input_ids, attention_mask).logits.squeeze().argmax()
    return output_label

def correct_print(asn,total):
    print("Attack Success")
    print('ASR: ',asn / total)


def attack(model,orig_test,tokenizer,ATTACK_ITER,task_name):
    stopwords_list = stopwords.words('english')
    correct = 0
    correct_samples=[]
    for sentence, label in tqdm(orig_test):
        output_label = get_output_label(sentence,tokenizer,model)
        if output_label == label:
            correct += 1.
            if label==1:
                correct_samples.append((sentence,1))

    print('Orig Acc: ', correct / len(orig_test))


    if len(correct_samples)>1000:
        correct_samples = random.sample(correct_samples, 1000)

    action_list = [insert_irrelevant, delete_char, swap_char, sub_char]  # insert_space, insert_period
    asn = 0
    total=0
    success_list=[]
    querytime=0
    print(ATTACK_ITER)
    if task_name =="enron" or task_name == "assassin":
        begsl = 10
        endsl = 30
        attack_batchsize=6
        attack_totalnum =30
        epoch=2
        ch=">"
        attack_ori_only=True
    elif task_name =="CGFake" or task_name == "amazon_lb":
        print("fakereview")
        begsl = 3
        endsl = 8
        attack_totalnum =100
        attack_batchsize=6
        epoch=3
        ch="up"
        attack_ori_only=True
    elif task_name =="LUN" or task_name == "satnews":
        # print("fakereview")
        begsl = 5
        endsl = 30
        attack_totalnum =100
        attack_batchsize=8
        epoch=3
        ch="reuters"
        attack_ori_only=True
    elif task_name =="jigsaw" or task_name == "HSOL":
        # print("fakereview")
        begsl = 0
        endsl = 180
        attack_totalnum =180
        attack_batchsize=4
        epoch=3
        ch="peace"
        attack_ori_only= True


    elif task_name =="FAS" or task_name == "EDENCE":
        begsl = 0
        endsl = 20
        attack_totalnum =100
        attack_batchsize=3
        epoch=3
        # ch="public"
        ch = "any"
        # ch ="the"
        attack_ori_only= True


    for sentence, label in tqdm(correct_samples):
        querytime_each=0
        total=total+1
        sentence = sentence.lower()
        orig_sentence=sentence
        orig_output_label = get_output_label(sentence,tokenizer,model)
        if orig_output_label == label:
            # Attack algorithm flow
            flag = False
            maxsl = len(sentence.split(' '))-1
            sentence = insert_long_distractor(sentence,begsl,endsl,ch)
            # print(sentence)
            output_label = get_output_label(sentence,tokenizer,model)
            querytime = querytime+1
            querytime_each=querytime_each+1
            if output_label != label:
                asn+=1
                correct_print(asn,total)
                success_list.append([orig_sentence,sentence,querytime_each])
                continue

            cur_iter = 0
            cache_pos_li = []


            # if sentence_len<500:  #200
            #     maxsl = sentence_len-1
            # else:
            #     maxsl = 500

            if attack_ori_only ==False:
                maxsl = len(sentence.split(' '))-1


            if maxsl >500:
                maxsl =500


            word_pos = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]

            while cur_iter < ATTACK_ITER and len(cache_pos_li) <maxsl*0.98 and  len(cache_pos_li)<attack_totalnum:
                words_li = sentence.split(' ')
                sentence_len = len(words_li)
                for p in range(attack_batchsize):
                    if attack_ori_only:
                        word_pos[p] = random.choice(list(range(begsl, begsl+maxsl)))
                    else:
                        word_pos[p] = random.choice(list(range(0, maxsl)))

                    
                    


                for e in range(attack_batchsize*epoch+int(0.05*len(cache_pos_li))): #2  #5

                    # cur=cur_iter
                    for i in range(attack_batchsize): #3  #2

                        # print("-------")
                        # print(random.seed)

                        if word_pos[i] in cache_pos_li: continue
                        # else: cache_pos_li.append(word_pos)

                        target_word = words_li[word_pos[i]]

                        if target_word in stopwords_list: continue



                        action = random.choice(action_list)
                        new_word = action(target_word)
                        # if new_word is None and action == sub_char:
                        #     cache_pos_li.remove(word_pos)
                        #     continue
                        if new_word is None:
                            continue


                        words_li = words_li[:word_pos[i]] + [new_word] + words_li[word_pos[i]+1:]
                        sentence = ' '.join(words_li)

                    output_label = get_output_label(sentence,tokenizer,model)
                    querytime = querytime+1
                    querytime_each=querytime_each+1
                    if output_label != label:
                        asn+=1
                        # print("----")
                        correct_print(asn,total)
                        success_list.append([orig_sentence,sentence,querytime_each])
                        flag=True
                        break
                for p in range(attack_batchsize):
                    cache_pos_li.append(word_pos[p])
                if flag==True:
                    break

        # rec_times=cur+rec_times
        else:
            # rec_times=cur+rec_times
            continue
        
    print(querytime)
    print(asn)
    name = ['origin','attack',"querytime"]
    save_data = pd.DataFrame(columns=name,data=success_list)
    save_data.to_csv(base_path+"output/"+task_name+'rocket.csv')
    
    print('ASR: ',asn / len(correct_samples))





def pipe(dataset_name,attack_iter):

    bert_type = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(bert_type)
    model_path = base_path+"/model/"
    model = torch.load(model_path+dataset_name)
    if torch.cuda.is_available():
        model = nn.DataParallel(model.cuda())


    


    orig_test = load_data(dataset_name,"dev")


    ATTACK_ITER = attack_iter
    attack(model,orig_test,tokenizer,ATTACK_ITER,dataset_name)
    print("-"*10+dataset_name+"-"*10)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='amazon_lb', type=str)
    args = parser.parse_args()

    pipe(args.name,550)

