import pandas as pd
import numpy as np
import csv
from collections import defaultdict as ddict
from evaluation import evaluation  
import pdb


def load_data(path):
    users = pd.read_csv(path+'/users.tsv', sep='\t')
    test_users = pd.read_csv(path+'/test_users.tsv', sep='\t')
    apps = pd.read_csv(path+'/apps.tsv', sep='\t')
    user_history = pd.read_csv(path+'/user_history.tsv', sep='\t')
    datamap = {
        'users':users,
        'test_users':test_users,
        'apps':apps,
        'user_history':user_history
    }
    return datamap

def data_info(data):
    print 'number of rows:'
    print len(data.index)
    print 'number of columns:'
    print len(data.columns)

def user_based_prediction(number_of_prediction):
    path = "./dataset/" # The directory that the data files are in
    user_based_prediction = pd.read_csv(path + "test_app_project.tsv", sep='\t',header=0)
    test_users =  user_based_prediction['UserID'].apply(str)
    test_users = test_users.unique();#get distinct test users
    test_users = test_users.tolist();#convert nparray to list
    #user-job table
    user_jobs = ddict(list)
    #job-user table
    job_users = ddict(list)
    predicted_user_jobs = ddict(lambda: ddict(list))
    print "-----              Recording job information              -----"
    job_info = {}
    with open(path + "splitjobs/jobs1.tsv", "r") as infile:
        reader = csv.reader(infile, delimiter="\t", 
        quoting=csv.QUOTE_NONE, quotechar="")
        reader.next() # burn the header
        for line in reader:
            (Jobid, WindowId, Title, Description, Requirements, City, State, 
            Country, Zip5, StartDate, EndDate) = line
            job_info[str(Jobid)] = [int(WindowId), State, City, 0]
            # The terminal zero is for an application count

    print "-----                Scanning applications                -----"
    with open(path + "train_app_project.tsv") as infile:
        reader = csv.reader(infile, delimiter="\t")
        reader.next() # burn the header
        for line in reader:
            (index, userId, split, applicationDate, jobId) = line
            #if windowID == 2: break
            user_jobs[str(userId)].append(str(jobId))
            job_users[str(jobId)].append(str(userId))

    print "-----     Predicting applications using Jaccard index     -----"
    with open(path + "test_app_project.tsv", "r") as infile:
        reader = csv.reader(infile, delimiter="\t", 
        quoting=csv.QUOTE_NONE, quotechar="")
        reader.next() # burn the header
        count = 0
        for line in reader:
            (index, userId, split, application, job_id) = line
            userId = str(userId)
            job_id = str(job_id)
            if userId not in test_users:
                continue
            test_users.remove(userId)

            #user-based collaborative filter algorithm
            for job_id in user_jobs[userId]:
                #job_id : jobs that the test user apply
                for user_id1 in job_users[job_id]:
                    #users that apply the same job
                    union_size = len(set(user_jobs[userId] + user_jobs[user_id1]))#size of the jobs 
                    for job_id1 in user_jobs[user_id1]:
                        #jobs that the similar user apply(has applied jobs in common with the oroginal user)
                        if job_id1 in user_jobs[userId]: break #if the user already apply that job, break
                        if predicted_user_jobs[userId].has_key(job_id1):
                            #if it is already in the prediction set, add 1/(user A 'and' user B) to the predicted_rank of the user
                            predicted_user_jobs[userId][job_id1] += 1.0/union_size 
                        else:
                            #if not, init the predicted_user_jobs
                            predicted_user_jobs[userId][job_id1] = 1.0/union_size
            print '-' * (count % 100)
            count += 1

    print "-----          Sorting user-based CF ranked jobs          -----"
    predicted_job_tuples = ddict(list)
    for user_id in predicted_user_jobs.keys():
       for job_id, count in predicted_user_jobs[user_id].items():
          predicted_job_tuples[user_id].append((job_id, count))
       predicted_job_tuples[user_id].sort(key=lambda x: x[1])
       predicted_job_tuples[user_id].reverse()
    #pdb.set_trace()
    '''
    #information based recommendation
    print "-----         Adding jobs on based on popularity          -----"
    top_city_jobs = ddict(lambda: ddict(list))
    top_state_jobs = ddict(list)
    for (job_id, (window, State, City, count)) in job_info.items():
        top_city_jobs[State][City].append((job_id, count))
        top_state_jobs[State].append((job_id, count))
    
    for state in top_city_jobs:
        for city in top_city_jobs[state]:
            top_city_jobs[state][city].sort(key=lambda x: x[1])
            top_city_jobs[state][city].reverse()
    for state in top_state_jobs:
        top_state_jobs[state].sort(key=lambda x: x[1])
        top_state_jobs[state].reverse()
        '''

    print "-----                Making predictions                   -----"
    user_based_prediction = pd.read_csv(path + "test_app_project.tsv", sep='\t',header=0)
    test_users =  user_based_prediction['UserID'].apply(str)
    test_users = test_users.unique();#get distinct test users

    
    with open(path + "users.tsv", "r") as infile:
        reader = csv.reader(infile, delimiter="\t", 
        quoting=csv.QUOTE_NONE, quotechar="")
        reader.next() # burn the header
        with open("user_based_prediction.csv", "w") as outfile:
            outfile.write("UserId,JobIds\n")
            for line in reader:
                (UserId, WindowId, Split, City, State, Country, ZipCode,
                DegreeType, Major, GraduationDate, WorkHistoryCount,
                TotalYearsExperience, CurrentlyEmployed, ManagedOthers,
                ManagedHowMany) = line

                

                if UserId in test_users:
                    #print '-' * (count % 100)
                    #count += 1    
                    top_jobs = predicted_job_tuples[UserId]
                    #if predicted user application is less than n, fill it with popular jobs in the same city or state 
                    '''
                    if len(top_jobs) < number_of_prediction:
                       top_jobs += top_city_jobs[State][City]
                    if len(top_jobs) < number_of_prediction:
                        top_jobs += top_state_jobs[State]
                        '''
                    top_jobs = top_jobs[0:number_of_prediction]
                    outfile.write(str(UserId) + "," + " ".join([x[0] for x in top_jobs]) + "\n")
    #return user_job score
    model = predicted_user_jobs
    return model       



if __name__ == '__main__':
    datamap = load_data('./dataset')
    users = datamap['users']
    user_history = datamap['user_history']
    #data_info(users)
    #data_info(datamap['user_history'])
    #data_info(datamap['apps'])
    model = user_based_prediction(5)
    evaluation(model,5)