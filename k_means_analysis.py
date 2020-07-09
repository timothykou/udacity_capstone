import etl_data

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt  
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from joblib import dump, load

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def create_demo_df(profile_df):
    '''
    Args:
    profile_df - cleaned dataframe containing user data mapped to the offers that they interacted with

    Returns:
    demo_df - dataframe containing only demographic user data with columns  ['F', 'M', 'O', 'age', 'member_time_days', 'income'] with numeric values
    '''
    demo_df =  profile_df[['gender', 'age', 'became_member_on', 'income']].copy()

    # add member_time_days    
    demo_df['member_start_date'] = demo_df['became_member_on'].apply(lambda start_date_int: pd.Timestamp(str(start_date_int)).date())
    last_member_joined_date =  pd.Timestamp(str(demo_df['became_member_on'].max())).date()
    demo_df['member_timedelta'] = last_member_joined_date - demo_df['member_start_date']
    demo_df['member_time_days']  = demo_df['member_timedelta'].apply(lambda num_days: num_days.days)
    
    # dummy 'gender' into 'F', 'M', 'O'
    demo_df = pd.concat([demo_df, pd.get_dummies(demo_df['gender'])], axis=1)

    return demo_df[['F', 'M', 'O', 'age', 'member_time_days', 'income']]


def k_means_cluster(demo_df):
    '''
    Args:
    demo_df - dataframe containing only demographic user data with columns  ['F', 'M', 'O', 'age', 'member_time_days', 'income']

    Returns:
    k  - number of clusters
    demo_df - added column with labels
    model.labels_ - list of labels
    '''
    kmeans_model_fp = os.path.abspath('') + '/Models/kmeans.joblib'
    try:
        model = load(kmeans_model_fp)
        print(f'Loaded K-Means Clustering Model from {kmeans_model_fp}')
        k = len(model.cluster_centers_)

    except:
        # use StandardScaler() standardize columns with non-binary integer values - age, member_time_days, income
        features_to_scale = ['age', 'member_time_days', 'income']
        df_to_scale = demo_df[features_to_scale]
        ct = ColumnTransformer([
            ('name', StandardScaler(), features_to_scale)
        ], remainder='passthrough')
        scaled = ct.fit_transform(df_to_scale)
        demo_df[features_to_scale] = scaled    
        
        X = demo_df.to_numpy()

        # Group users by demographic using K-Means clustering
            # Find optimal # clusters using elbow method: 4
        inertias = [] 
        num_clusters = range(1,10) 

        for k in num_clusters: 
            print(f'Trying {k} clusters...')
            model = KMeans(n_clusters=k, init='k-means++').fit(X) 
            model.fit(X)     
        
            inertias.append(model.inertia_) 

        # visualize elbow method - Shows that the best number of clusters is: 4
        plt.plot(num_clusters, inertias, 'bx-') 
        plt.xlabel('# Clusters') 
        plt.ylabel('WCSS') 
        plt.title('The Elbow Method using WCSS') 
        plt.show()    
        
        # get cluster labels
        k = 4
        print('Fitting model...')
        model = KMeans(n_clusters=k, init='k-means++', random_state=1).fit(X)

        try:
            dump(model, kmeans_model_fp)
            print(f'Saved model to {kmeans_model_fp}')
        except:
            print(f'Could not save model to {kmeans_model_fp}')

    # label user by cluster
    demo_df['kmeans_cluster'] = model.labels_
    return k, demo_df, model.labels_



def count_offer_interactions(profile_offer_df):
    # get data for clusters and offer interactions only
    cluster_offer_col_list = ['kmeans_cluster']
    for i in range(1,11):
        cluster_offer_col_list.append(i)
    cluster_offer_df = profile_offer_df[cluster_offer_col_list]
    
    offer_dict = {}
    for cluster_num in range(1, num_clusters+1):
        offer_dict[cluster_num] = {}
        cluster = cluster_offer_df[cluster_offer_df['kmeans_cluster']==cluster_num]
        for offer_num in range(1, 11):
            cluster_offer = pd.DataFrame(cluster[offer_num])
            offer_interactions_df = cluster_offer.apply(lambda row: pd.Series(row[offer_num], index=['received_only', 'viewed_completed', 'viewed_only', 'completed_only']), axis=1)
            d = offer_interactions_df.sum().to_dict()
            offer_dict[cluster_num][offer_num] = d
            offer_dict[cluster_num][offer_num]['received'] = offer_interactions_df.sum().sum()
    print(offer_dict)
    
    offer_interaction_counts_df = pd.DataFrame(offer_dict)

    return offer_interaction_counts_df


def subplots_offer_interaction_pcts_by_cluster(offer_interaction_counts_df):
    # plot the %'s of offers viewed and completed, viewed only, and completed only
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles = ('Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4')   #  change if number of clusters changes
    )
    for cluster_num in range(1, len(offer_interaction_counts_df.columns)+1):
        cluster = pd.DataFrame(offer_interaction_counts_df[cluster_num])
        # expand the offer_interaction_counts from dictionaries into dataframe
        expanded_df = cluster.apply(lambda row: pd.Series(row[cluster_num]), axis=1, result_type='expand')
        cluster = cluster.merge(expanded_df, how='outer', left_index=True, right_index=True)

        cluster['received_pct'] = 100
        cluster['received_only_pct'] = 100 * (cluster['received_only'] / cluster['received'])
        cluster['viewed_completed_pct'] = 100 * (cluster['viewed_completed'] / cluster['received'])
        cluster['viewed_only_pct'] = 100 * (cluster['viewed_only'] / cluster['received'])
        cluster['completed_only_pct'] = 100 * (cluster['completed_only'] / cluster['received'])

        for chart_type in ['viewed_completed_pct', 'received_only_pct', 'viewed_only_pct', 'completed_only_pct']:
            fig.add_trace(
                go.Bar(
                    x = cluster.index.tolist(),
                    y= cluster[chart_type].tolist(),
                    name=chart_type,
                    
                ),
                row=cluster_num,
                col=1,
            )
    fig.update_layout(title_text=f'Offer Interaction %\'s for each Demographic Cluster')
    return fig


def visualize_irr(offer_interaction_counts_df):
    # Visualize incremental response rate by cluster and offer ########################################################################################################################
        # 1 plot
    all_clusters_viewed_completed = np.zeros(10)
    all_cluster_completed_only =  np.zeros(10)
    all_clusters_offer_group_count = np.zeros(10)
    all_clusters_control_group_count = np.zeros(10)
    irr_fig = go.Figure()
    irr_dict  = {}
    # analyze which offers to give to each group
    for cluster_num in range(1, len(offer_interaction_counts_df.columns)+1):
        cluster = pd.DataFrame(offer_interaction_counts_df[cluster_num])
        # expand the offer_interaction_counts from dictionaries into dataframe
        expanded_df = cluster.apply(lambda row: pd.Series(row[cluster_num]), axis=1, result_type='expand')
        cluster = cluster.merge(expanded_df, how='outer', left_index=True, right_index=True)

        cluster['received_pct'] = 100
        cluster['viewed_completed_pct'] = 100 * (cluster['viewed_completed'] / cluster['received'])
        cluster['viewed_only_pct'] = 100 * (cluster['viewed_only'] / cluster['received'])
        cluster['completed_only_pct'] = 100 * (cluster['completed_only'] / cluster['received'])

        all_clusters_viewed_completed += cluster['viewed_completed'].to_numpy()
        all_cluster_completed_only += cluster['completed_only'].to_numpy()
        # Incremental response rate: purchased in offer group / # in offer group) - (# purchased in control group / # in control group)
            # The 'offer' group: all users who received and viewed the offer
        offer_group_count = cluster['viewed_completed'] + cluster['viewed_only']
        all_clusters_offer_group_count += offer_group_count.to_numpy()
            # The 'control' group: all users who received the offer but did not view it
        control_group_count = cluster['received'] - offer_group_count
        all_clusters_control_group_count += control_group_count.to_numpy()
        cluster['incremental_response_rate'] =  (cluster['viewed_completed'] / offer_group_count) - (cluster['completed_only'] / control_group_count)

        irr_dict[cluster_num] = {}
        for offer_num in cluster.index.tolist():
            irr_dict[cluster_num][offer_num]  = cluster.loc[offer_num, 'incremental_response_rate']

        # plot incremental response rate bars by offer for each cluster
        irr_fig.add_trace(
            go.Bar(
                x = cluster.index.tolist(),
                y = cluster['incremental_response_rate'].tolist(),
                name=f'Cluster {cluster_num}',
            )
        )
    # plot irr across clusters
    all_clusters_irr = (all_clusters_viewed_completed / all_clusters_offer_group_count) - (all_cluster_completed_only / all_clusters_control_group_count)
    irr_fig.add_trace(
        go.Bar(
            x=cluster.index.tolist(),
            y=all_clusters_irr,
            name=f'All Profiles'
        )
    )
    irr_fig.update_layout(title='Incremental Response Rate by Cluster for each Offer #', barmode='group', xaxis={'tickmode':'linear'})
    return irr_fig, irr_dict


def remove_negative_irr_offers(row, offer_num, irr_dict):
        cluster_num = row['kmeans_cluster']
        if irr_dict[cluster_num][offer_num] < 0:
            return np.nan
        else:
            return row[offer_num]


if __name__ == "__main__":
    # load data
    profile_offer_df, portfolio_df =  etl_data.setup_data()

    # load data frame with counts of offer interactions by cluster 
    offer_interaction_counts_fp = os.path.abspath('') + '/Data/offer_interaction_counts_kmeans.pkl'

    profile_offer_df_fp = os.path.abspath('') + '/Data/profile_offer_df.pkl'
    try:
        offer_interaction_counts_df  = pd.read_pickle(offer_interaction_counts_fp)
        profile_offer_df = pd.read_pickle(profile_offer_df_fp)
        # get demographic data as separate dataframe for clustering
        demo_df = create_demo_df(profile_offer_df)

        # use k-means clustering to group customers
        num_clusters, clustered_demo_df, labels = k_means_cluster(demo_df)
        profile_offer_df['kmeans_cluster'] = labels+1     

    # create df if it does not exist
    except:    
        # get demographic data as separate dataframe for clustering
        demo_df = create_demo_df(profile_offer_df)

        # use k-means clustering to group customers
        num_clusters, clustered_demo_df, labels = k_means_cluster(demo_df)
        profile_offer_df['kmeans_cluster'] = labels+1       

        #  count offer interactions by cluster 
        print('counting offer interactions by cluster')
        offer_interaction_counts_df = count_offer_interactions(profile_offer_df)       

        # save offer_interaction_counts
        print(f'Saving offer_interaction_counts_df to {offer_interaction_counts_fp}')
        try:
            offer_interaction_counts_df.to_pickle(offer_interaction_counts_fp)
            print(f'Successfully saved offer_interaction_counts_df to {offer_interaction_counts_fp}')
        except:
            print(f'Could not save offer_interaction_counts_df to {offer_interaction_counts_fp}')

    print('Running visualizations...\n')
    # Visualize demographic data by cluster ########################################################################################################################
    # add member_time_days
    profile_offer_df['member_start_date'] = profile_offer_df['became_member_on'].apply(lambda start_date_int: pd.Timestamp(str(start_date_int)).date())
    last_member_joined_date =  pd.Timestamp(str(profile_offer_df['became_member_on'].max())).date()
    profile_offer_df['member_timedelta'] = last_member_joined_date - profile_offer_df['member_start_date']
    profile_offer_df['member_time_days']  = profile_offer_df['member_timedelta'].apply(lambda num_days: num_days.days)
    
    # get demographic data for each cluster
    cluster_demo_data = profile_offer_df[['gender', 'age', 'member_time_days', 'income', 'kmeans_cluster']]
    # compare each demographic metric by cluster
    demo_subplots = make_subplots(
        rows=1, cols=5,
        subplot_titles = ('# Ppl', 'Gender %', 'Avg Age', 'Membership', 'Avg Income')   
    )
    subplot_col=1
    # add the number in each cluster
    len_clusters = []
    cluster_nums = []
    for cluster_num in cluster_demo_data['kmeans_cluster'].unique():
        cluster = cluster_demo_data[cluster_demo_data['kmeans_cluster']==cluster_num]
        len_clusters.append(len(cluster))
        cluster_nums.append(cluster_num)
    demo_subplots.add_trace(
        go.Bar(
            x=cluster_nums,
            y=len_clusters,
            name='# people in each cluster'
        ),
        row=1,
        col=subplot_col,
    )
    subplot_col += 1
    for col_name in cluster_demo_data.drop('kmeans_cluster', axis=1).columns:
        if col_name == 'gender':
            f_pct = []
            m_pct = []
            o_pct = []
            cluster_nums = []
            # get numbers for each gen
            for cluster_num in cluster_demo_data['kmeans_cluster'].unique():
                cluster = cluster_demo_data[cluster_demo_data['kmeans_cluster']==cluster_num]
                f_pct.append(len(cluster[cluster['gender']=='F'])/len(cluster))
                m_pct.append(len(cluster[cluster['gender']=='M'])/len(cluster))
                o_pct.append(len(cluster[cluster['gender']=='O'])/len(cluster)) 
                cluster_nums.append(cluster_num)
            demo_subplots.add_trace(
                go.Bar(
                    x=cluster_nums,
                    y=f_pct,
                    name='% F'
                ),
                row=1,
                col=subplot_col,
            )
            demo_subplots.add_trace(
                go.Bar(
                    x=cluster_nums,
                    y=m_pct,
                    name='% M'
                ),
                row=1,
                col=subplot_col,
            )
            demo_subplots.add_trace(
                go.Bar(
                    x=cluster_nums,
                    y=o_pct,
                    name='% O'
                ),
                row=1,
                col=subplot_col,
            )
            demo_subplots.update_layout(barmode='group')
            subplot_col+=1
        else:
            cluster_averages = []
            cluster_nums = []
            for cluster_num in cluster_demo_data['kmeans_cluster'].unique():
                cluster = cluster_demo_data[cluster_demo_data['kmeans_cluster']==cluster_num]
                cluster_averages.append(cluster[col_name].mean())
                cluster_nums.append(cluster_num)
            demo_subplots.add_trace(
                go.Bar(
                    x=cluster_nums,
                    y=cluster_averages,
                    name=col_name
                ),
                row=1,
                col=subplot_col,
            )
            subplot_col+=1
    demo_subplots.show()

    # Visualize offer interaction data by cluster ########################################################################################################################
        # 4 subplots - 1 cluster per plot
    fig = subplots_offer_interaction_pcts_by_cluster(offer_interaction_counts_df)
    fig.show()

    # Visualize incremental response rate by cluster and offer ########################################################################################################################
    print(f'Visualizing IRR for sample data')
    irr_fig, irr_dict = visualize_irr(offer_interaction_counts_df)
    irr_fig.show()

    # Compare the sample data IRR to an alternative IRR: in the scenario where clusters with negative IRR are not given that offer
    # create alternate profile_offer_df - where users are only given offers where their cluster has a positive irr
    alt_profile_offer_df =  profile_offer_df.copy()    
    for i in range(1, 11):
        alt_profile_offer_df[i] =  alt_profile_offer_df.apply(remove_negative_irr_offers, args=[i, irr_dict], axis=1)
    print(alt_profile_offer_df)

    # load or create alternative offer interaction counts in this scenario
    alt_offer_interaction_counts_fp = os.path.abspath('') + '/Data/alt_offer_interaction_counts_kmeans.pkl'
    try:
        alt_offer_interaction_counts_df  = pd.read_pickle(alt_offer_interaction_counts_fp)
        print(f'Loading alt_offer_interaction_counts_df from {alt_offer_interaction_counts_fp}')
    except:
        # create alternate offer_interaction_counts_df
        print(f'Creating alt_offeration_counts_df')
        alt_offer_interaction_counts_df = count_offer_interactions(alt_profile_offer_df)       
    
        # save alternate offer_interaction_counts    
        print(f'Saving alt_offer_interaction_counts_df to {alt_offer_interaction_counts_fp}')
        try:
            alt_offer_interaction_counts_df.to_pickle(alt_offer_interaction_counts_fp)
            print(f'Successfully saved alt_offer_interaction_counts_df to {alt_offer_interaction_counts_fp}')
        except:
            print(f'Could not save alt_offer_interaction_counts_df to {alt_offer_interaction_counts_fp}')
    
    # Visualize incremental response rate by cluster and offer ########################################################################################################################
    print(f'Visualizing IRR for alternative data based on cluster IRR')
    irr_fig, irr_dict = visualize_irr(alt_offer_interaction_counts_df)
    irr_fig.show()
    

    

