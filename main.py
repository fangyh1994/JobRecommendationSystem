import pandas as pd
import numpy as np
import csv
from collections import defaultdict as ddict
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

def user_based_prediction():
    wd = "./dataset/" # The directory that the data files are in

    #user-job table
    user_jobs = ddict(list)
    #job-user table
    job_users = ddict(list)
    predicted_user_jobs = ddict(lambda: ddict(list))
    print "Recording job locations..."
    job_info = {}
    with open(wd + "splitjobs/jobs1.tsv", "r") as infile:
        reader = csv.reader(infile, delimiter="\t", 
        quoting=csv.QUOTE_NONE, quotechar="")
        reader.next() # burn the header
        for line in reader:
            (Jobid, WindowId, Title, Description, Requirements, City, State, 
            Country, Zip5, StartDate, EndDate) = line
            job_info[str(Jobid)] = [int(WindowId), State, City, 0]
            # The terminal zero is for an application count

    print "Counting applications..."
    with open(wd + "appstrain.tsv") as infile:
        reader = csv.reader(infile, delimiter="\t")
        reader.next() # burn the header
        for line in reader:
            (UserId, WindowID, Split, ApplicationDate, JobId) = line
            if WindowID == 2: break
            user_jobs[UserId].append(JobId)
            job_users[JobId].append(UserId)

    print "Finding similar jobs..."
    with open(wd + "users.tsv", "r") as infile:
        reader = csv.reader(infile, delimiter="\t", 
        quoting=csv.QUOTE_NONE, quotechar="")
        reader.next() # burn the header
        for line in reader:
            (UserId, WindowId, Split, City, State, Country, ZipCode,
            DegreeType, Major, GraduationDate, WorkHistoryCount,
            TotalYearsExperience, CurrentlyEmployed, ManagedOthers,
            ManagedHowMany) = line
            if Split == "Train":
                continue

            #user-based collaborative filter algorithm
            for job_id in user_jobs[UserId]:
               for user_id1 in job_users[job_id]:
                  union_size = len(set(user_jobs[UserId] + user_jobs[user_id1]))
                  for job_id1 in user_jobs[user_id1]:
                     if job_id1 in user_jobs[UserId]: break
                     if predicted_user_jobs[UserId].has_key(job_id1):
                        predicted_user_jobs[UserId][job_id1] += 1.0/union_size
                     else:
                        predicted_user_jobs[UserId][job_id1] = 1.0/union_size

    print "Sorting collaborative filtering jobs..."
    predicted_job_tuples = ddict(list)
    for user_id in predicted_user_jobs.keys():
       for job_id, count in predicted_user_jobs[user_id].items():
          predicted_job_tuples[user_id].append((job_id, count))
       predicted_job_tuples[user_id].sort(key=lambda x: x[1])
       predicted_job_tuples[user_id].reverse()

    #information based recommendation
    print "Sorting jobs on based on popularity..."
    top_city_jobs = ddict(lambda: ddict(lambda: ddict(list)))
    top_state_jobs = ddict(lambda: ddict(list))
    for (job_id, (window, State, City, count)) in job_info.items():
        top_city_jobs[window][State][City].append((job_id, count))
        top_state_jobs[window][State].append((job_id, count))
    for window in [1]:
        for state in top_city_jobs[window]:
            for city in top_city_jobs[window][state]:
                top_city_jobs[window][state][city].sort(key=lambda x: x[1])
                top_city_jobs[window][state][city].reverse()
        for state in top_state_jobs[window]:
            top_state_jobs[window][state].sort(key=lambda x: x[1])
            top_state_jobs[window][state].reverse()

    print "Making predictions..."
    with open(wd + "users.tsv", "r") as infile:
        reader = csv.reader(infile, delimiter="\t", 
        quoting=csv.QUOTE_NONE, quotechar="")
        reader.next() # burn the header
        with open("popular_jobs2.csv", "w") as outfile:
            outfile.write("UserId, JobIds\n")
            for line in reader:
                (UserId, WindowId, Split, City, State, Country, ZipCode,
                DegreeType, Major, GraduationDate, WorkHistoryCount,
                TotalYearsExperience, CurrentlyEmployed, ManagedOthers,
                ManagedHowMany) = line
                if Split == "Train":
                    continue
                top_jobs = predicted_job_tuples[UserId]
                #if predicted user application is less than 150, fill it with popular jobs in the same city or state 
                if len(top_jobs) < 150:
                   top_jobs += top_city_jobs[int(WindowId)][State][City]
                if len(top_jobs) < 150:
                    top_jobs += top_state_jobs[int(WindowId)][State]
                top_jobs = top_jobs[0:150]
                outfile.write(str(UserId) + "," + " ".join([x[0] for x in top_jobs]) + "\n")


if __name__ == '__main__':
    datamap = load_data('./dataset')
    users = datamap['users']
    user_history = datamap['user_history']
    data_info(users)
    #data_info(datamap['user_history'])
    #data_info(datamap['apps'])
    user_based_prediction()