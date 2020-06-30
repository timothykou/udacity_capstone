import pandas as pd
import numpy as np
import os
import math
import json
import plotly.graph_objects as go


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


def clean_transcript_df(transcript):
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


def check_for_event(row, event_type, transcript_df):
    try:
        # # if looking for completed offers, first check if offer has been viewed
        # if event_type == 'offer_completed' and np.isnan(row['offer_viewed']):
        #     return np.nan
        # look for relevant events
        events = transcript_df[(transcript_df['person']==row['person']) & (transcript_df['time']>=row['time'])
                            & (transcript_df['time']<=row['end_time']) & (transcript_df['event']==event_type)]
        if len(events)>0:
            if event_type == 'transaction':
                # multiple transactions are possible in the duration of the offer - add event indexes to tuple
                return tuple(events.index.tolist())
            else:
                # return the event index of the relevant event
                return events.index[0]
        else:
            return np.nan
    except:
        print(row)
        print(event_type)
        return np.nan


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
        

def create_offer_map(transcript_df, portfolio):
    '''
    Args:
    transcript_df - cleaned transcript DataFrame
    portfolio - cleaned portfolio DataFrame
    
    Returns:
    offer_map (df) - DataFrame of offers received by each user with data on whether the offer was viewed and completed, and any transactions and total spend made in that offer period
    '''

    # Break into chunks of 500 rows for speed
    start_row = 0
    
    # initialize offer_map dataframe
    offer_map = pd.DataFrame()

    while start_row <= len(transcript_df):
        print(f'Working on rows {start_row}-{start_row+500} out of {len(transcript_df)}')

        # get offers_received in transcript_df
        df = transcript_df[start_row:(start_row+500)].copy()
        df = df[df['event']=='offer_received']
        if len(df) >  0:        
            # add on offer data from portfolio df
            df = df.merge(portfolio, how='left', left_on='offer_id', right_on='id').rename(columns={'duration':'duration_days'})
            df.drop(columns=['id'], inplace=True)
            df['duration_hours'] = df['duration_days'] * 24
            df['end_time']  = df['time'] + df['duration_hours']

            # for each offer received, check other events within the offer period
            for event in ['offer_viewed', 'transaction', 'offer_completed']:    # offer_viewed must be before offer_completed
                print(f'Looking for event: {event}')
                df[event] = df.apply(check_for_event, args=[event, transcript_df], axis=1)
            
            # get total spend for all offers that had related transactions
            df['transaction_total'] = df['transaction'].apply(get_transactions_total)

        # add to offer_map
        offer_map = pd.concat([offer_map, df])
        start_row += 500
    
    return offer_map


def save_df(df, filepath):
    try:
        df.to_pickle(fp)
        print(f'Successfully saved df to {fp}')
        return
    except:
        print(f'Could not save df to {fp}')
    return 


def count_events(user_id, offer_id, col_type, offer_map):
    user_offers = offer_map[(offer_map['person']==user_id) & (offer_map['offer_id']==offer_id)]
    if len(user_offers) == 0:
        return 0
    if col_type == 'transaction_total':
        return user_offers[col_type].sum()
    if col_type == 'offer_received':
        return len(user_offers)
    else:
        return user_offers[col_type].count()
    return


def create_profile_df(profile, portfolio, offer_map):
    '''
    Args:
    profile - 'profile' DataFrame loaded from /Data/profile.json
    portfolio - cleaned portfolio DataFrame

    Returns:
    profile_df (df) - DataFrame containing each user's profile data and their events (viewed, completed, transactions, transaction_total) that occurred during offer periods
    '''
    df = profile.copy()

    # Loop through each offer 
    for i in range(len(portfolio)):
        offer_id = portfolio.loc[i, 'id']
        columns_to_add = event_types
        columns_to_add.append('transaction_total')

        # Add columns to show events for each offer - one column for each offer's events: [offers_received, offer_viewed, offers_completed, transactions]
        for col_type in columns_to_add:
            offer_num = portfolio.loc[i, 'offer_num']
            col_name = f'{offer_num}_{col_type}'
            # add column with data for this offer and event for each customer
            print(f'Adding column: {col_name}')
            df[col_name] = df['id'].apply(count_events, args=[offer_id, col_type, offer_map])
    
    return df


def create_demo_df(profile_df):
    '''
    Args:
    profile_df - cleaned dataframe containing user data mapped to the transactions and offers that they interacted with

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


def setup_data():
    '''
    Args:
    None

    Returns:
    profile_df - cleaned dataframe containing user data mapped to the transactions and offers that they interacted with
    demo_df - dataframe containing only demographic user data with columns  ['F', 'M', 'O', 'age', 'member_time_days', 'income']
    '''
    portfolio, profile, transcript = import_files()

    # Clean transcript data
    transcript_df = clean_transcript_df(transcript)
    event_types = transcript_df['event'].unique().tolist()  

    # Clean portfolio data
    portfolio_df = clean_portfolio_df(portfolio)

    # Map offer interactions (views, completions, and transactions within offer period) to each offer for each user
    
    profile_df_fp = os.path.abspath('') + '/Data/profile_df.pkl'
    # if profile_df already exists, load it
    try:
        profile_df = pd.read_pickle(profile_df_fp)
        print(f'Loaded profile_df from {profile_df_fp}\n')
    except:
    # otherwise create profile_df
        # if offer_map already exists, load it
        offer_map_fp = os.path.abspath('') + '/Data/offer_map.pkl'
        try:
            offer_map = pd.read_pickle(offer_map_fp)
            print(f'Loaded offer_map from {offer_map_fp}\n')
        # otherwise, create offer_map
        except:
            print(f'Creating offer_map\n')
            offer_map = create_offer_map(transcript_df, portfolio_df)
            print(f'Successfully created offer_map\n')
            save_df(offer_map, offer_map_fp)

        print(f'Creating profile_df\n')
        profile_df = create_profile_df(profile, portfolio_df, offer_map)
        print(f'Successfully created profile_df\n')
        save_df(profile_df, profile_df_fp)

    # Clean profile_df
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
    
    demo_df = create_demo_df(profile_df)
    
    return profile_df, demo_df, portfolio_df

    
    