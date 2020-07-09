import pandas as pd
import numpy as np
import os
import math
import json
import plotly.graph_objects as go
import sys


def import_files():
    # read in the json files
    portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
    profile = pd.read_json('data/profile.json', orient='records', lines=True)
    transcript = pd.read_json('data/transcript.json', orient='records', lines=True)

    return portfolio, profile, transcript


def rename_key_w_underscore(value_dict):
    renamed_dict = {}
    for k, v in value_dict.items():
        renamed_dict[k.replace(' ', '_')] = v
    return renamed_dict 


def get_value_val(value_dict, value_key):
    try:
        return value_dict[value_key]
    except:
        return np.nan


def clean_transcript_df(transcript, portfolio_df):
    '''
    Args:
    transcript - 'transcript' DataFrame loaded from /Data/transcript.json
    
    Returns:
    transcript_df - cleaned transcript data
    '''
    transcript_df = transcript.copy()
    
    # rename all events with underscores: (e.g. 'offer id' value keys as 'offer_id')
    transcript_df['event'] = transcript_df['event'].apply(lambda event: event.replace(' ',  '_'))

    # replace spaces in 'value' col keys with underscore
    transcript_df['value'] = transcript_df['value'].apply(rename_key_w_underscore)
    # Separating 'value' col of dicts into other columns (keys: 'offer_id', 'amount', 'reward') - NaN if that event does not have this key
    for value_key in ['offer_id', 'amount', 'reward']:
        transcript_df[value_key] = transcript_df['value'].apply(get_value_val, args=[value_key])

    # remove the 'value' column afterward
    transcript_df.drop(columns='value', inplace=True)

    # add the offer difficulty, reward, and expiration time
    offer_data = portfolio_df[['id', 'difficulty', 'reward', 'duration', 'offer_num']]
    transcript_df = transcript_df.merge(offer_data, how='left', left_on='offer_id', right_on='id')
    transcript_df['expiration'] = transcript_df['time'] + (24 * transcript_df['duration'])

    transcript_df.drop_duplicates(inplace=True)
    return transcript_df


def clean_portfolio_df(portfolio):
    '''
    Args:
    portfolio - 'portfolio' DataFrame loaded from /Data/portfolio.json
    
    Returns:
    portfolio_df - cleaned portfolio data
    '''
    portfolio_df = portfolio.copy()

    # Adding offer_num column to map offer ID to assist with visualization
    portfolio_df['offer_num'] = portfolio_df.index + 1
    return portfolio_df


def clean_profile_df(profile):
    profile_df = profile.copy()
    # remove users with missing demographic data
        # found 2175 users with no gender data: all users are missing income data and have age of 118 - remove these users 
    profile_df = profile_df[pd.isnull(profile_df['gender'])==False]
    # check remaining users for valid data
        # income - found no users with missing income data. Fill mising values with mean income
    if (pd.isna(profile_df['income']).sum() > 0):
        profile_df['income'].fillna(profile_df['income'].mean())
        # became_member_on - found no users with missing membership start dates. drop row if this is missing
    if (pd.isna(profile_df['became_member_on']).sum() > 0):
        profile_df.dropna(subset='became_member_on', inplace=True)
        # age - found no users with missing ages. fill with mean age if missing
    if (pd.isna(profile_df['age']).sum() > 0):
        profile_df['age'].fillna(profile_df['age'].mean())
    
    return profile_df


def chart_event_type_counts(transcript_df, portfolio):
    # Distribution of Offers in Sample Data
    received_count = pd.DataFrame(transcript_df[transcript_df['event']=='offer_received'].groupby('offer_id')['event'].count())
    received_count = received_count.rename(columns={'event':'count'})
    received_count = received_count.merge(portfolio[['id', 'offer_num']], how='left', left_on='offer_id', right_on='id')

    viewed_count = pd.DataFrame(transcript_df[transcript_df['event']=='offer_viewed'].groupby('offer_id')['event'].count())
    viewed_count = viewed_count.rename(columns={'event':'count'})
    viewed_count = viewed_count.merge(portfolio[['id', 'offer_num']], how='left', left_on='offer_id', right_on='id')

    completed_count = pd.DataFrame(transcript_df[transcript_df['event']=='offer_completed'].groupby('offer_id')['event'].count())
    completed_count = completed_count.rename(columns={'event':'count'})
    completed_count = completed_count.merge(portfolio[['id', 'offer_num']], how='left', left_on='offer_id', right_on='id')

    data =  [
        go.Bar(x=received_count['offer_num'], y=received_count['count'], name='offers received'),
        go.Bar(x=viewed_count['offer_num'], y=viewed_count['count'], name='offers viewed'),
        go.Bar(x=completed_count['offer_num'], y=completed_count['count'], name='offers completed'),
    ]
    layout = dict(
        title='Number of Total Offers Received, Viewed, and Completed by Offer #',
        xaxis=dict(
            title='Offer #',
            tickmode='linear')    
    )
    fig = go.Figure(data, layout)
    fig.show()


def get_transactions_total(transaction_tuple):
    if isinstance(transaction_tuple, tuple):
        total = 0
        for transaction_ix in list(transaction_tuple):
            try:
                total += transcript_df.loc[transaction_ix, 'amount']
            except:
                print(f'Amount for {transaction_ix} could not be added')
                continue
        return total
    else:
        return np.nan


def save_df(df, fp):
    try:
        df.to_pickle(fp)
        print(f'Successfully saved df to {fp}')
        return
    except:
        print(f'Could not save df to {fp}')
    return 


def create_demo_df(profile_df):
    '''
    Args:
    profile_df - cleaned dataframe containing user data mapped to the offers that they interacted with

    Returns:
    demo_df - dataframe containing only demographic user data with columns  ['F', 'M', 'O', 'age', 'member_time_days', 'income']
    '''
    demo_df =  profile_df[['gender', 'age', 'became_member_on', 'income']].copy()
    
    demo_df['member_start_date'] = demo_df['became_member_on'].apply(lambda start_date_int: pd.Timestamp(str(start_date_int)).date())
    last_member_joined_date =  pd.Timestamp(str(demo_df['became_member_on'].max())).date()
    demo_df['member_timedelta'] = last_member_joined_date - demo_df['member_start_date']
    demo_df['member_time_days']  = demo_df['member_timedelta'].apply(lambda num_days: num_days.days)
    
    # dummy 'gender' into 'F', 'M', 'O'
    demo_df = pd.concat([demo_df, pd.get_dummies(demo_df['gender'])], axis=1)

    print(demo_df[['F', 'M', 'O', 'age', 'member_time_days', 'income']])
    return demo_df[['F', 'M', 'O', 'age', 'member_time_days', 'income']]



def offer_interactions(row, person_transcript):
    '''
    person_transcript - transcript of events for one user
    '''
    # filter for interactions with this specific offer (views and completions within the timeframe)
    offer_interactions = person_transcript[(person_transcript['id']==row['offer_id']) & (person_transcript['time']>=row['time']) & (person_transcript['time']<=row['expiration']) 
                            & (person_transcript['event'].isin(['offer_viewed', 'offer_completed']))].reset_index()
    interactions = {'received_only':0, 'viewed_completed': 0, 'viewed_only': 0, 'completed_only': 0}
    # check if the user interacted with this offer
    if len(offer_interactions) > 0:
        # check if the offer was completed without being viewed
        if offer_interactions.loc[0, 'event']=='offer_completed':
            interactions['completed_only'] = 1
        # if viewed first - check if coimpleted
        elif offer_interactions.loc[0, 'event']=='offer_viewed':
            if len(offer_interactions) > 1:
                if offer_interactions.loc[1, 'event']=='offer_completed':
                    interactions['viewed_completed'] = 1
            else:
                interactions['viewed_only'] = 1
        else:
            print('found other type of event:', offer_interactions)
    else:
        interactions['received_only'] =  1
    try:
        s = pd.Series(interactions,  index=['received_only', 'viewed_completed', 'viewed_only', 'completed_only'])
    except:
        print('failed:', interactions)
    return s


def person_offer_data(row, transcript_df):
    # get received offers for each person
    person_transcript = transcript_df[transcript_df['person']==row['id']]
    offers = person_transcript[person_transcript['event']=='offer_received']

    # retrieves bools for whether each received offer was only viewed, only completed, or viewed and completed - then converts to 1s and 0s
    interactions_df = offers.apply(offer_interactions, args=[person_transcript,], axis=1)

    # sum interaction type by offer number
    offer_interactions_df = offers.merge(interactions_df, how='left', left_index=True, right_index=True)
    if len(offer_interactions_df) == 0:
        # if user did not receive any offers, return blank Series
        return pd.Series()
    offer_interaction_counts = (offer_interactions_df.groupby('offer_num')[['received_only', 'viewed_completed', 'viewed_only', 'completed_only']].sum())
    
    # return as list of 1s and  0s for the number of offer interactions coresponding to:  ['received_only', 'viewed_completed', 'viewed_only', 'completed_only']
    offer_interaction_lists_df = pd.Series()
    for offer_num in offer_interaction_counts.index.tolist():
        offer_interaction_lists_df[offer_num] = offer_interaction_counts.loc[offer_num, :].tolist()
    return offer_interaction_lists_df
    

def setup_data():
    '''
    Args:
    None

    Returns:
    profile_df - cleaned dataframe containing user data mapped to the transactions and offers that they interacted with
    demo_df - dataframe containing only demographic user data with columns  ['F', 'M', 'O', 'age', 'member_time_days', 'income']
    '''
    portfolio, profile, transcript = import_files()

    # Clean portfolio data
    portfolio_df = clean_portfolio_df(portfolio)
    
    # Clean transcript data
    transcript_df = clean_transcript_df(transcript, portfolio_df)
    event_types = transcript_df['event'].unique().tolist()  
    
    # visualize events in transcript by count
    chart_event_type_counts(transcript_df,  portfolio_df)

    # Clean profile data
    profile_df = clean_profile_df(profile)

    # Map counts of offer interactions (views, completions, and transactions within offer period) to each offer for each user
    profile_offer_df_fp = os.path.abspath('') + '/Data/profile_offer_df.pkl'
    # if profile_offer_df already exists, load it
    try:
        profile_offer_df = pd.read_pickle(profile_offer_df_fp)
        print(f'Loaded profile_offer_df from {profile_offer_df_fp}\n')
    
    # otherwise create profile_offer_df
    except:
        print(f'Creating profile_offer_df\n')
        offer_interactions = profile_df.apply(person_offer_data, args=[transcript_df,], axis=1, result_type='expand')
        profile_offer_df = profile_df.merge(offer_interactions, how='left', left_index=True, right_index=True)
        print(f'Successfully created profile_offer_df\n')
        print(profile_offer_df)
        save_df(profile_offer_df, profile_offer_df_fp)
    
    return profile_offer_df, portfolio_df
    

if __name__ == "__main__":
    setup_data()