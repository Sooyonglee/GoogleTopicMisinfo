#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 10:28:11 2024

@author: tdn897
"""

import pandas as pd
import os
import random
import numpy as np

def calculate_test_statistic(row):
    abs_llm_diff = abs(row['predict_label']['Female'] - row['predict_label']['Male'])
    abs_human_diff = abs(row['true_label']['Female'] - row['true_label']['Male'])
    return abs_llm_diff - abs_human_diff

# Returns the empirical test statistic for each topic
def shape_topic_level_test_statistic(merged):
    # Pivot the original data to get true_label and predict_label for each gender for each claim
    pivot_data = merged_data.pivot_table(index=['topic', 'claim'], columns='gender', values=['true_label', 'predict_label']).reset_index()

    # Apply the function to each row
    pivot_data['test_statistic'] = pivot_data.apply(calculate_test_statistic, axis=1)

    # Group by topic and calculate the mean test statistic
    topic_level_statistic = pivot_data.groupby('topic')['test_statistic'].mean().reset_index()
    return topic_level_statistic

# Creating the desired dictionary
# Structure: {topic: {claim: {'human male': [], 'human female': [], 'LLM male': [], 'LLM female': []}}}

def create_dictionary(merged_data):
    topic_dict = {}
    for index, row in merged_data.iterrows():
        topic = row['topic']
        claim = row['claim']
        gender = row['gender']
        true_label = row['true_label']
        predict_label = row['predict_label']
    
        # Initialize nested dictionary if topic not in dictionary
        if topic not in topic_dict:
            topic_dict[topic] = {}
    
        # Initialize claim dictionary if claim not in topic dictionary
        if claim not in topic_dict[topic]:
            topic_dict[topic][claim] = {'human_male': [], 'human_female': [], 'LLM_male': [], 'LLM_female': []}
    
        # Classify and append the annotations to the respective list
        if gender == 'Male':
            topic_dict[topic][claim]['human_male'].append(true_label)
            topic_dict[topic][claim]['LLM_male'].append(predict_label)
        elif gender == 'Female':
            topic_dict[topic][claim]['human_female'].append(true_label)
            topic_dict[topic][claim]['LLM_female'].append(predict_label)
    return topic_dict



def run_bootstrap(B, topic_dict):
    results = []
    for topic in list(topic_dict.keys()):
        print('Topic: ' + topic + '\n\n\n')
        for claim in topic_dict[topic]:
            # Bootstrap resampling from joint distribution by gender.
            for b in range(0, B):
                n1 = len(topic_dict[topic][claim]['human_male'])
                n2 = len(topic_dict[topic][claim]['human_female'])
                n3 = len(topic_dict[topic][claim]['LLM_male'])
                n4 = len(topic_dict[topic][claim]['LLM_female'])
                # Calculate observed test statistic
                ## Combine male and female responses across (human vs LLM) condition 
                ## This should make a more conservative test than joining all data
                combined_male = topic_dict[topic][claim]['human_male'] + topic_dict[topic][claim]['LLM_male'] 
                combined_female = topic_dict[topic][claim]['human_female'] + topic_dict[topic][claim]['LLM_female']
                resampled_n1 = np.mean(random.choices(combined_male, k=n1))
                resampled_n2 = np.mean(random.choices(combined_female, k=n2))
                resampled_n3 = np.mean(random.choices(combined_male, k=n3))
                resampled_n4 = np.mean(random.choices(combined_female, k=n4))
                bootstrap_test_stat = np.abs(resampled_n4 - resampled_n3) - np.abs(resampled_n2 - resampled_n1)
                results.append([topic, claim, bootstrap_test_stat, b])
                
    return results


###########
#### MAIN
###########
B = 20000

# Change working directory
os.chdir('/Users/tdn897/Desktop/GoogleTopicMisinfo/')
# Load the datasets
groupharm_data = pd.read_csv('data/gpt-35/groupharm-conditional-results-prompt2.csv')
ground_truth_data = pd.read_csv('data/GroundTruthPreExperiment.csv')
# Merging the datasets on the 'claim' column
merged_data = pd.merge(groupharm_data, ground_truth_data, left_on='claim', right_on='source', how='inner')
# Create observed test statistic for each claim
topic_level_statistic = shape_topic_level_test_statistic(merged = merged_data)
# create topic_dict
topic_dict = create_dictionary(merged_data = merged_data)
# Run Bootstrapping procedure
results = run_bootstrap(B = B, topic_dict = topic_dict)
# Bootstrap results at topic, claim, bootstrap rep level
raw_results = pd.DataFrame(results, columns=['topic', 'claim', 'bootstrap_test_stat', 'rep'])
# aggregate bootstrap results to topic, bootstrap rep level
rep_level = raw_results.groupby(['topic', 'rep'])['bootstrap_test_stat'].mean().reset_index()
# merge data with observed test statistics by topic
results = pd.merge(rep_level, topic_level_statistic, on='topic')
# Calculate score - 1 if observed is greater than bootstrap result, 0 otherwise
results['score'] = np.where(results['test_statistic']>results['bootstrap_test_stat'], 1, 0)
# Calculate proportion of reps where observed is greater than the bootstrap average
topic_results = results.groupby(['topic'])['score'].mean()
