import etl_data

import pandas as pd
# import numpy as np 
import matplotlib.pyplot as plt  
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# from sklearn import metrics 
# from scipy.spatial.distance import cdist 


def k_means_cluster(demo_df):
    '''
    Args:
    demo_df - dataframe containing only demographic user data with columns  ['F', 'M', 'O', 'age', 'member_time_days', 'income']

    Returns:
    TODO
    '''

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
    '''
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
    '''

    
    # get cluster labels
    k = 4
    model = KMeans(n_clusters=k, init='k-means++', random_state=1).fit(X)

    # label user by cluster
    demo_df['kmeans_cluster'] = model.labels_
    return k, demo_df, model.labels_


if __name__ == "__main__":
    profile_df, demo_df, portfolio_df =  etl_data.setup_data()

    num_clusters, clustered_demo_df, labels = k_means_cluster(demo_df)
    profile_df['kmeans_cluster'] = labels

    event_types = ['offer_received',  'offer_viewed', 'offer_completed', 'transaction', 'transaction_total']
    
    # store data by offer by cluster
    offer_dict = {}
    for i in range(1,11):
        offer_dict[i] = {}
        reward_amt = portfolio_df.loc[(i-1), 'reward']

        # get columns for this offer
        offer_columns = []
        # get columns to keep
        for event in event_types:
            col_name = f'{i}_{event}'
            offer_columns.append(col_name)
        
        # check how each cluster interacted with this offer
        for cluster_num in range(1, num_clusters+1):
            offer_dict[i][cluster_num] = {}

            # filter for users in this cluster
            cluster = profile_df[profile_df['kmeans_cluster'] == (cluster_num-1)]
            
            received_col = f'{i}_offer_received'
            viewed_col = f'{i}_offer_viewed'
            completed_col = f'{i}_offer_completed'
            transaction_col = f'{i}_transaction'
            transaction_total_col = f'{i}_transaction_total'
            received_col = f'{i}_offer_received'
            received_col = f'{i}_offer_received'

            # filter each cluster for users that received each offer
            offered = cluster[offer_columns][cluster[received_col]>0]

            # save data for each cluster and offer: 
            #   num_received; num_viewed; pct_viewed; num_completed; pct_completed; num_transactions; pct_transactions; sum_transaction_total; avg_spend_per_received; net_incremental_revenue
            offer_dict[i][cluster_num]['num_received'] = offered[received_col].sum()
            
            offer_dict[i][cluster_num]['num_viewed'] = offered[viewed_col].sum()
            offer_dict[i][cluster_num]['pct_viewed'] = offered[viewed_col].sum() / offered[received_col].sum()

            offer_dict[i][cluster_num]['num_completed'] = offered[completed_col].sum()
            offer_dict[i][cluster_num]['pct_completed'] = offered[completed_col].sum() / offered[received_col].sum()

            offer_dict[i][cluster_num]['num_transactions'] = offered[transaction_col].sum()
            offer_dict[i][cluster_num]['pct_transactions'] = offered[transaction_col].sum() / offered[received_col].sum()
            
            offer_dict[i][cluster_num]['sum_transaction_total'] = offered[transaction_total_col].sum()
            offer_dict[i][cluster_num]['avg_spend_per_received'] = offered[transaction_total_col].sum() / offered[received_col].sum()
            
            # net_incremental_revenue = (total spend from users that viewed offer - (offer reward * num times completed)) - total spend from users that did not view the offer
            transaction_total_no_view = offered[offered[viewed_col]==0][transaction_total_col].sum()
            offer_dict[i][cluster_num]['net_incremental_revenue'] = (offered[transaction_total_col].sum() - (reward_amt * offered[completed_col].sum())) - transaction_total_no_view
            
    # visualizations ############################################################
    
    # show the percent viewed by cluster for each offer
    
    # pct_completed_by_cluster = []
    # pct_transactions_by_cluster = []
    # avg_spend_per_received_by_cluster = []
    # net_incremental_revenue_by_cluster = []
    # for offer_num in range(1, 11):
    #     for cluster_num in range(1, num_clusters+1):
    #         # pct_viewed_by_cluster.append(offer_dict[offer_num][cluster_num]['pct_viewed'])
    #         pct_completed_by_cluster.append(offer_dict[offer_num][cluster_num]['pct_completed'])
    #         pct_transactions_by_cluster.append(offer_dict[offer_num][cluster_num]['pct_transactions'])
    #         avg_spend_per_received_by_cluster.append(offer_dict[offer_num][cluster_num]['avg_spend_per_received'])
    #         net_incremental_revenue_by_cluster.append(offer_dict[offer_num][cluster_num]['net_incremental_revenue'])

    # print(pct_viewed_by_cluster)
    # pct_completed_by_cluster = []
    # pct_transactions_by_cluster = []
    # avg_spend_per_received_by_cluster = []
    # net_incremental_revenue_by_cluster = []

    # plot bar charts of each cluster's interaction with each offer
        # 1 chart for each offer, with 1 bar for each cluster
    for chart_type in ['pct_viewed', 'pct_completed', 'pct_transactions',  'avg_spend_per_received', 'net_incremental_revenue']:
        fig = make_subplots(
            rows=1, cols=10,
            subplot_titles=('Offer 1', 'Offer 2', 'Offer 3', 'Offer 4', 'Offer 5', 'Offer 6', 'Offer 7', 'Offer 8', 'Offer 9', 'Offer 10')
        )
        for offer_num in range(1, 11):
            pct_viewed_by_cluster = []
            for cluster_num in range(1, num_clusters+1):
                pct_viewed_by_cluster.append(offer_dict[offer_num][cluster_num][chart_type])
            fig.add_trace(
                go.Bar(
                    x = list(range(1, cluster_num+1)),
                    y = pct_viewed_by_cluster,
                ),
                row=1,
                col=offer_num
            )
        fig.update_layout(height=500, width=1000,title_text=f'{chart_type} for each Demographic Cluster')
        fig.show()







    


# fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]),
#               row=1, col=1)

# fig.add_trace(go.Scatter(x=[20, 30, 40], y=[50, 60, 70]),
#               row=1, col=2)

# fig.add_trace(go.Scatter(x=[300, 400, 500], y=[600, 700, 800]),
#               row=2, col=1)

# fig.add_trace(go.Scatter(x=[4000, 5000, 6000], y=[7000, 8000, 9000]),
#               row=2, col=2)

# fig.show()



    

    
    # chart_event_type_counts(transcript_df, portfolio_df)


# data =  [
#         go.Bar(x=received_count['offer_num'], y=received_count['count'], name='offers received'),
#         go.Bar(x=viewed_count['offer_num'], y=viewed_count['count'], name='offers viewed'),
#         go.Bar(x=completed_count['offer_num'], y=completed_count['count'], name='offers completed'),
#     ]
#     layout = dict(
#         title='Number of Total Offers Received, Viewed, and Completed by Offer #',
#         xaxis=dict(
#             title='Offer #',
#             tickmode='linear')    
#     )
#     fig = go.Figure(data, layout)
#     fig.show()


