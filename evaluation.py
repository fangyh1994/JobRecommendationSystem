import pandas as pd
import numpy as np
import csv
from collections import defaultdict as ddict
import random
import pdb


def evaluation(model, number_of_prediction):         
    path = "./dataset/"
    #both userid and jobId are converted to string object
    correct_predict_count = 0
    app_count = 0
    predict_count = 0

    user_jobs = ddict(list)
    with open(path + "test_app_project.tsv") as infile:
        reader = csv.reader(infile, delimiter="\t")
        reader.next() # burn the header
        for line in reader:
            (index, userId, split, applicationDate, jobId) = line
            #if WindowID == 2: break
            app_count += 1
            user_jobs[userId].append(jobId)

    predicts = pd.read_csv('./user_based_prediction.csv', sep=',', converters={'JobIds': lambda x: str(x), 'UserId': lambda x: str(x)})
    #print len(predicts.index)
    for index, row in predicts.iterrows():
        userId = row["UserId"]
        for job in row["JobIds"].split():
            predict_count += 1
            if job in user_jobs[userId]:
                #test code
                '''
                if correct_predict_count < 10:
                    print job
                    print user_jobs[userId]
                    '''
                #print job
                correct_predict_count += 1
    print "correctly predicted application number: " + str(correct_predict_count)
    print "total application number: " + str(app_count)
    print "total predicted number: " + str(predict_count)

    #calculate recall 5 and recall 10
    path = "./dataset/" # The directory that the data files are in
    user_based_prediction = pd.read_csv(path + "test_app_project.tsv", sep='\t',header=0)
    test_users =  user_based_prediction['UserID'].apply(str)
    #test_users = test_users.unique();#get distinct test users
    #test_users = test_users.tolist();#convert nparray to list
    test_jobs =  user_based_prediction['JobID'].apply(str)
    #test_jobs = test_users.unique();#get distinct test users
    #test_jobs = test_users.tolist();#convert nparray to list

    job_list = pd.read_csv(path + "train_app_project.tsv", sep='\t',header=0)
    jobId_list =  job_list['UserID'].apply(str)
    jobId_list = jobId_list.unique();#get distinct test users
    jobId_list = jobId_list.tolist();#convert nparray to list

    #user model tuple(jobid, score) 
    predicted_job_tuples = ddict(list)
    hit = 0
    total_count = 0
    for user_id, job_id in zip(test_users, test_jobs):
        random_jobId_list = random.sample(jobId_list, 100)#create 100 random jobs
        score = 0

        if user_id in model.keys() and job_id in model[user_id].keys():
            #pdb.set_trace()
            score = model[user_id][job_id]
        if job_id not in random_jobId_list:
            predicted_job_tuples[user_id].append((job_id, score))
            #add current job score to the list of tuple
        
        for random_job_id in random_jobId_list:
            #add random_job and score into the tuple
            if user_id in model.keys() and random_job_id in model[user_id].keys():
                predicted_job_tuples[user_id].append((random_job_id, model[user_id][random_job_id]))
            else:
                predicted_job_tuples[user_id].append((random_job_id,0))
        total_count = total_count + 1 #total number of test users application
        predicted_job_tuples[user_id].sort(key=lambda x: x[1])
        #sort the tuple by score
        predicted_job_tuples[user_id].reverse()#greatest to smallest
        res_tuple = predicted_job_tuples[user_id][0:number_of_prediction]
        #topN jobId
        #print res_tuple
        for job, score in res_tuple:
            #Recall@topN: whether the jobId is in the predicted jobs
            if job == job_id:
                print "hit"
                hit += 1
        print '-' * (total_count % 100)
        #count += 1

    print ("Recall at " + str(number_of_prediction) + ":")
    recall = hit/float(total_count)
    print recall
    #print predicted_job_tuples    

    #iterate through all test users
    

    #print random_jobId_list
    '''
    if user in test_user_list:
        test_user_list.remove(user)
    if user in 
    predicted_job_tuples[user_id].append(())
    '''
            
if __name__ == '__main__':
    model = {'72':{'480634':0.01,'564184':0.002},'395':{'123032':0.01,'345':0.002}}
    number_of_prediction = 10
    evaluation(model,number_of_prediction)
    

