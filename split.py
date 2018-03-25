import numpy as np
import scipy
import pandas as pd
import sklearn
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from datetime import datetime #datetime type
from sklearn.feature_extraction.text import TfidfVectorizer

jobs1_df = pd.read_csv('dataset/splitjobs/jobs1.tsv', delimiter="\t", na_filter=False, error_bad_lines=False)
print(jobs1_df.head(5))

app_df = pd.read_csv('dataset/apps.tsv', delimiter="\t")
app_df = app_df[app_df['WindowID'] == 1] # only make recommendation within window1, cause the dataset is too large
app_df = app_df.drop('WindowID', axis=1) # delete the column 'WindowID'.
#print(app_df) # 0 ~ 353581 applications, 63412 users sent at least 1 application.

# To avoid user cold-start problems, keep in the dataset only for users with at least 5 applications.
users_app_count_df = app_df.groupby(['UserID', 'JobID']).size().groupby('UserID').size()
users_with_enough_apps_df = users_app_count_df[users_app_count_df >= 5].reset_index()[['UserID']]
#print(users_with_enough_apps_df) # 18254 users with at least 5 interactions

apps_from_selected_users_df = app_df.merge(users_with_enough_apps_df, how='right', left_on='UserID', right_on='UserID')
#print(apps_from_selected_users_df.describe()) # 266675 applications from selected users
#print(apps_from_selected_users_df.dtypes) # ApplicationDate's type is object


# select the test users in window1 who had applied jobs in the training time of window1 (i.e. before 4/10/2012 12:00:00 AM) 
test_users = pd.read_csv('dataset/test_users.tsv', delimiter="\t")
test_users = test_users[test_users['WindowID'] == 1]
#print(test_users_window1) #5419 test users in window1
train_end_time = datetime(2012, 4, 10, 0, 0, 0) #2012-04-10 00:00:00   Train End / Test Start of window1
test_end_time = datetime(2012, 4, 14, 0, 0, 0) #2012-04-14 00:00:00   Test End of window1

#print(datetime.strptime('4/2/2012 10:36:43 PM', '%m/%d/%Y %H:%M:%S %p')) # a simple test of datetime converting
app_df['ApplicationDate'] = pd.to_datetime(app_df['ApplicationDate']) #convert string to datetime
train_app_df = app_df[(app_df['ApplicationDate'] < train_end_time) & (app_df['JobID'].isin(jobs1_df['JobID']))] # only keep the applications in the training time, and the jobs should be assigned to window1 
#print(train_app_df) #315439 applications in the training time of window1

# check the applications made by the test users in the training time
test_users_app_in_training_time_df = train_app_df.loc[train_app_df['UserID'].isin(test_users['UserID'])] 
# select rows whose column value in the set of test users
# Note: do not use merge method, because it will cause many fields with values of NaN or NaT

#print(test_users_app_in_training_time_df) #38583 applications are made by the test users in the training time

test_users_app_count_df = test_users_app_in_training_time_df.groupby(['UserID', 'JobID']).size().groupby('UserID').size()
#print(test_users_app_count_df) # 2379 test users made applications in the training time of window1
test_users_with_enough_apps_df = test_users_app_count_df[test_users_app_count_df >= 5].reset_index()
export_target_users = test_users_with_enough_apps_df.drop(test_users_with_enough_apps_df.columns[1], axis=1)
#print(export_target_users) # [users] 1486 test users made >= 5 applications in the training time, which are the target users for our content-based approach.
#export_target_users.to_csv('target_users_project.tsv', sep='\t') # write dataframes of target users to csv file


total_test_users_training_applications = test_users_with_enough_apps_df[test_users_with_enough_apps_df.columns[1]].sum() # [applications] get the sum of the second column of test_users_with_enough_apps_df in order to get the number of samples for training 
#print(total_test_users_training_applications) # 36674 applications can be used as training data


# Now, let us preprocess the test data
test_app_df = app_df[(app_df['ApplicationDate'] > train_end_time) & (app_df['JobID'].isin(jobs1_df['JobID']))]
test_users_app_in_test_time_df = test_app_df.loc[test_app_df['UserID'].isin(test_users['UserID'])].reset_index() # [applications] applications made by the test users in the test time
#print(test_users_app_in_test_time_df) # 11166 applications can be tested. 
# Note: The time of application in test_users_app_in_test_time_df does not have to be before the end of test time of window 1

# export the test jobs which ends after 2012-04-10 00:00:00 (the end of training)
# jobs1_df['EndDate'] = pd.to_datetime(jobs1_df['EndDate']) # convert the format of 'EndDate' to DateTime
# test_jobs_df = jobs1_df[jobs1_df['EndDate'] > train_end_time].reset_index()
# test_jobs_df = test_jobs_df.drop('WindowID', axis=1)
# test_jobs_df = test_jobs_df.drop('Zip5', axis=1)
# test_jobs_df = test_jobs_df.drop('Country', axis=1)
# test_jobs_df = test_jobs_df.drop('StartDate', axis=1)
# test_jobs_df = test_jobs_df.drop('EndDate', axis=1)
# print(test_jobs_df)
#test_jobs_df.to_csv('test_jobs_project.tsv', sep='\t') # write dataframes of test jobs to csv file
train_app_df.to_csv('train_app_project.tsv', sep='\t')
test_app_df.to_csv('test_app_project.tsv', sep='\t')