#  Starbucks Capstone Challenge - Udacity Data Scientist Nanodegree Capstone Project

## Table of contents

1. [Installation](#installation)
2. [Project Overview](#project-overview)
3. [File Descriptions](#file-descriptions)
4. [Running this Project](#running-this-project)
5. [Summary](#summary)
6. [Acknowledgements](#acknowledgements)    
  
<br>  <br />  

## Installation

#### Dependencies
This project uses the following packages:
    
    Python (>=3.7.4)

    numpy==1.19.0
    pandas==1.0.5
    plotly==4.8.1
    scikit-learn==0.23.1
    matplotlib==3.2.2
<br>  <br />  

## Project Overview 
This assignment is the Data Scientist Capstone project for the Udacity Data Scientist Nanodegree Program:
    Udacity's Data Scientist Nanodegree curriculum: https://www.udacity.com/course/data-scientist-nanodegree--nd025  
The blog post discussing the results from this assignment can be found here: https://medium.com/@timothydkou/customer-segmentation-for-analyzing-interaction-behavior-with-promotions-by-demographic-d01a94dd5374?source=friends_link&sk=d47a5de8405234d7365e8a3781dd7d6a

The Starbucks Capstone Challenge, chosen for this project, is to combine the provided simulated datasets containing transaction, demographic, and offer data to determine which mobile application users respond best to which offer types.


The challenge comes with the following datasets:
    * portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
    * profile.json - demographic data for each customer
    * transcript.json - records for transactions, offers received, offers viewed, and offers completed

<br>  <br />  

## File Descriptions

```
├── Data
│   ├── clustered_demo_df.pkl
│   ├── offer_interaction_counts_gmm.pkl
│   ├── offer_interaction_counts_kmeans.pkl
│   ├── offer_interaction_counts_kmeans.pkl
│   ├── portfolio.json
│   ├── profile.json
│   ├── profile_offer_df.pkl
│   └── transcript.json
├── Models
│   ├── gmm.joblib
│   └── kmeans.joblib
├── etl_data.py
├── gmm_analysis.py
├── k_means_analysis.py
├── README.md
└── requirements.txt
```

- Models directory - stores K-means clustering and Gaussian Mixture models that were used
- Data directory  - store necessary data for analysis after transformation
- etl_data.py - contains data processing and transformation steps
- gmm_analysis.py - contains code for analysis and visualization using Gaussian Mixture Model to segment demographics
- k_means_analysis.py - contains code for analysis and visualization using K-Means Clustering to segment demographics
<br>  <br />  

## Running this project
- etl_data.py should be used for pre-processing and transformation steps
- k_means_analysis.py can be run to obtain the results discussed in the Medium blog post.


<br>  <br />  
## Summary
In this Starbucks Capstone Challenge, I used K-Means Clustering to segment customer profiles by demographic data. Then, I processed and transformed a log of events related to transactions and promotions delivered through the mobile app. I analyzed how different demographic clusters may interact with offers differently, and made some recommendations. Finally, I measured the incremental response rate for each offer from each cluster and found some differences in IRR between clusters for the same offers.

Analysis of the results is discussed in the Medium blog post.

<br>  <br />  
## Acknowledgements
Author: Tim Kou (https://github.com/timothykou)

Link to Medium blog post on this subject: https://medium.com/@timothydkou/customer-segmentation-for-analyzing-interaction-behavior-with-promotions-by-demographic-d01a94dd5374?source=friends_link&sk=d47a5de8405234d7365e8a3781dd7d6a
