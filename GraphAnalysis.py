import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from lifelines.statistics import logrank_test
from itertools import combinations
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
import matplotlib.cm as cm;
import itertools
from sklearn.cluster import DBSCAN

class GraphAnalysis:
    def __init__(self, EXTRACTER):
        self.extracter = EXTRACTER
        self.process();
        
    def process(self):
        latent_features_list = list(self.extracter.latent_feat_dict.values())
        patient_list = list(self.extracter.latent_feat_dict.keys())
        latentF = torch.stack(latent_features_list, dim=0)
        self.latentF = np.squeeze(latentF.numpy())
        
        self.pIDs = patient_list
        self.df = pd.DataFrame(columns=['PC1','PC2','tX','tY','groups'],index=self.pIDs)
        # self.pca_tsne();
        self.clnc_df = pd.read_csv('fdata/hnsc2/survival.hnsc_data.csv').set_index('PatientID')
        self.df = self.df.join(self.clnc_df)
        

    
    def pca_tsne(self):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.latentF)
        self.df['PC1'] = X_pca[:,0];
        self.df['PC2'] = X_pca[:,1];
        tsne = TSNE(n_components=2)
        X_tsne = tsne.fit_transform(self.latentF)
        self.df['tX'] = X_tsne[:,0];
        self.df['tY'] = X_tsne[:,1];
        
        
    def find_optimal_clusters(self, min_clusters=2, max_clusters=11,save_path='./fdata/results_830/kelbow'):
        model = KMeans(random_state=42)
        visualizer = KElbowVisualizer(model, k=(min_clusters, max_clusters))
        visualizer.fit(self.latentF)
        visualizer.show()

        fig = visualizer.ax.get_figure()
        fig.savefig(save_path + ".png", dpi=150)
        fig.savefig(save_path + ".jpeg", format="jpeg", dpi=150)

        
#         visualizer.savefig(save_path + ".png", dpi=300)  # Adjust the DPI as needed
#         visualizer.savefig(save_path + ".jpeg", format="jpeg", dpi=300) 
            
        self.optimal_clusters = visualizer.elbow_value_    
        

    def cluster_data(self):
        if self.optimal_clusters is None:
            raise ValueError("Please run 'find_optimal_clusters' method before clustering the data.")
        kmeans = KMeans(n_clusters=self.optimal_clusters, random_state=0).fit(self.latentF)
        self.labels = kmeans.labels_
        self.df['groups'] = self.labels 
        self.generate_color_list_based_on_median_survival()
        
    def cluster_data2(self,kclust):
        kmeans = KMeans(n_clusters=kclust, random_state=0).fit(self.latentF)
        self.labels = kmeans.labels_
        self.df['groups'] = self.labels 
        self.generate_color_list_based_on_median_survival()

    def visualize_clusters(self):
        plt.figure(figsize=(20,8))
        plt.subplot(1,2,1)
        sns.scatterplot(data=self.df, x='PC1', y='PC2', hue='groups', palette=self.color_list)

        plt.subplot(1,2,2)
        sns.scatterplot(data=self.df, x='tX', y='tY', hue='groups', palette=self.color_list)
        
    def save_visualize_clusters(self):
        plt.figure(figsize=(10,8))
        sns.scatterplot(data=self.df, x='PC1', y='PC2', hue='groups', palette=self.color_list)
        plt.savefig('fdata/results_830/temp_pca.jpeg',dpi=300)
        plt.savefig('fdata/results_830/temp_pca.png',dpi=300)
        
        plt.close();
        
        plt.figure(figsize=(10,8))
        sns.scatterplot(data=self.df, x='tX', y='tY', hue='groups', palette=self.color_list)
        plt.savefig('fdata/results_830/temp_tsne.jpeg',dpi=300)
        plt.savefig('fdata/results_830/temp_tsne.png',dpi=300)
         
        
    def map_group_to_color(group):
        return self.color_list[group]
        
    def generate_color_list_based_on_median_survival(self):
        # Compute median survival times for each group and sort
        groups = self.df['groups'].unique()
        median_survival_times = {group: self.df[self.df['groups'] == group]['Overall Survival (Months)'].median() for group in groups}
        sorted_groups = sorted(groups, key=median_survival_times.get, reverse=True)

        # Generate colors in VIBGYOR order
        vibgyor_colors = cm.rainbow(np.linspace(0, 1, len(groups)))

        # Assign colors based on sorted median survival times
        self.color_list = {group: color for group, color in zip(sorted_groups, vibgyor_colors)}
        
    def perform_log_rank_test(self, alpha=0.05):
        if self.df is None:
            raise ValueError("Please run 'cluster_data' or 'cluster_data2' method before performing log rank test.")
        
        groups = self.df['groups'].unique()
        significant_pairs = []

        print("Log Rank test for survival")
        print("Group 1 vs Group 2 : p_value")

        for pair in itertools.combinations(groups, 2):
            group_a = self.df[self.df['groups'] == pair[0]]
            group_b = self.df[self.df['groups'] == pair[1]]
            results = logrank_test(group_a['Overall Survival (Months)'],
                                   group_b['Overall Survival (Months)'],
                                   group_a['Overall Survival Status'],
                                   group_b['Overall Survival Status'])

            significance_marker = "****" if results.p_value < alpha else ""
            print(f"   {pair[0]}   vs   {pair[1]}   :  {results.p_value:.2f} {significance_marker}")

            if results.p_value < alpha:
                significant_pairs.append(pair)
        self.significant_pairs = significant_pairs
        return self.significant_pairs
    
    def generate_summary_table(self):
        groups = self.df['groups'].unique()
        summary_table = pd.DataFrame(columns=['Total number of patients', 'Alive', 'Deceased', 'Median survival time'], index=groups)

        for group in groups:
            group_data = self.df[self.df['groups'] == group]
            total_patients = len(group_data)
            alive = len(group_data[group_data['Overall Survival Status'] == 0])
            deceased = len(group_data[group_data['Overall Survival Status'] == 1])

            kmf = KaplanMeierFitter()
            kmf.fit(group_data['Overall Survival (Months)'], group_data['Overall Survival Status'])

            median_survival_time = kmf.median_survival_time_
            # confidence_interval = kmf.confidence_interval_['KM_estimate']

            summary_table.loc[group] = [total_patients, alive, deceased, median_survival_time]

        return summary_table
    
    def plot_kaplan_meier(self, plot_for_groups=True,name='temp_k5'):
        kmf = KaplanMeierFitter()

        plt.figure(figsize=(8, 6))
        plt.grid(False)
        if plot_for_groups:
            # Plot for different groups
            groups = sorted(self.df['groups'].unique())
            for i, group in enumerate(groups):
                group_data = self.df[self.df['groups'] == group]
                kmf.fit(group_data['Overall Survival (Months)'], group_data['Overall Survival Status'], label=f'Group {group}')
                kmf.plot(ci_show=False, linewidth=2, color=self.color_list[group])
            plt.title("Kaplan-Meier Curves for Each Group",
                      # fontweight='bold'
                     )
        else:
            # Plot for the whole data
            kmf.fit(self.df['Overall Survival (Months)'], self.df['Overall Survival Status'], label='All Data')
            kmf.plot(ci_show=False, linewidth=2, color='black')
            plt.title("Kaplan-Meier Curve for All Data",
                      # fontweight='bold'
                     )
            
        plt.gca().set_facecolor('#f5f5f5')
    
        # Customize the grid
        plt.grid(color='lightgrey', linestyle='-', linewidth=0.5)  
        # plt.grid(False)
        plt.xlabel("Overall Survival (Months)",fontweight='bold')
        plt.ylabel("Survival Probability",fontweight='bold')
        plt.legend()
        plt.savefig('fdata/results_830/{}_plan_meir.jpeg'.format(name),dpi=300);
        plt.savefig('fdata/results_830/{}_plan_meir.png'.format(name),dpi=300)
        plt.show()
        
    def club_two_groups(self, primary_group, secondary_group):
        # Update DataFrame
        self.df.loc[self.df['groups'] == secondary_group, 'groups'] = primary_group

        # Update Proceeding Group Numbers
        unique_groups = sorted(self.df['groups'].unique())
        mapping = {old: new for new, old in enumerate(unique_groups)}
        self.df['groups'] = self.df['groups'].map(mapping)

        # Recompute Color List
        self.generate_color_list_based_on_median_survival()

        # Recompute Summary
        self.summary_table = self.generate_summary_table()

        # Optional: Re-run Log-Rank Test
        # self.perform_log_rank_test()  # Uncomment if you want to re-run log-rank test

    def plot_median_survival_bar(self,name='temp_k5'):
        # Get the summary table
        summary_df = self.generate_summary_table()
        summary_df['group'] = summary_df.index

        # Handle the 'inf' values
        max_val = summary_df["Median survival time"].replace(np.inf, np.nan).max()
        summary_df["Display Median"] = summary_df["Median survival time"].replace(np.inf, max_val * 1.1)

        # Ensure summary_df is sorted by group index for correct color mapping
        summary_df = summary_df.sort_index()

        # Use the group colors
        colors = [self.color_list[group] for group in summary_df.index]

        # Adjust the figure dimensions for thinner bars
        num_groups = len(summary_df)
        plt.figure(figsize=(6, num_groups * 0.8))

        plt.grid(False)
        sns.barplot(data=summary_df, y='group', x="Display Median", palette=colors, orient="h", order=summary_df.index)
        plt.xlabel("Median Survival Time (Months)")
        plt.ylabel("Groups")
        plt.title("Median Survival Time by Group")

        plt.tight_layout()

        # Annotate the groups where median is 'inf'
        # for idx, group in enumerate(summary_df.index):
        #     if summary_df.loc[group, "Median survival time"] == np.inf:
        #         plt.text(max_val * 1.1, idx, "Not Reached", color='red', va='center')
        
        plt.savefig('fdata/results_830/{}_median_survival.jpeg'.format(name),dpi=300);
        plt.savefig('fdata/results_830/{}_median_survival.png'.format(name),dpi=300)

        plt.show()