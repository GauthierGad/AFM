#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 12:55:07 2023

@author: gauthier.gadouas
"""

import pandas as pd
import datetime
from pathlib import Path
import os
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

#list of maps to remove because coverage is not enough

# removed_maps = ["/Users/gauthier.gadouas/Desktop/Projet AFM/Data/Preprocessed_data//patient 1/tumeur/manip1/map 1 vertical-tip-position_processed-2018.06.21-15.30.06.xlsx",                
# "/Users/gauthier.gadouas/Desktop/Projet AFM/Data/Preprocessed_data//patient 1/tumeur/manip1/map 2 vertical-tip-position_processed-2018.06.21-15.33.26.xlsx",                
# "/Users/gauthier.gadouas/Desktop/Projet AFM/Data/Preprocessed_data//patient 13/tumeur/map 3 T13 map 3_processed-2019.07.02-15.09.29.xlsx",                                  
# "/Users/gauthier.gadouas/Desktop/Projet AFM/Data/Preprocessed_data//patient 15/metastase/epithelium/map 1 epitelium zone 2_processed-2020.02.14-14.46.23.xlsx",             
# "/Users/gauthier.gadouas/Desktop/Projet AFM/Data/Preprocessed_data//patient 2/tissu_sain/epithelium/map 1 2 N OCT epith 2_processed-2018.01.23-13.49.58.xlsx",              
# "/Users/gauthier.gadouas/Desktop/Projet AFM/Data/Preprocessed_data//patient 2/tissu_sain/matrice/map 2 2N OCT matrix 2_processed-2018.01.23-15.41.42.xlsx",                 
# "/Users/gauthier.gadouas/Desktop/Projet AFM/Data/Preprocessed_data//patient 2/tumeur/manip1/map 3 2T OCT 3_processed-2018.01.24-09.34.15.xlsx",                             
# "/Users/gauthier.gadouas/Desktop/Projet AFM/Data/Preprocessed_data//patient 2/tumeur/manip1/map 4 2T OCT 4_processed-2018.01.24-09.53.04.xlsx",                             
# "/Users/gauthier.gadouas/Desktop/Projet AFM/Data/Preprocessed_data//patient 2/tumeur/manip1/map 5 2T OCT 5_processed-2018.01.24-10.09.03.xlsx",                             
# "/Users/gauthier.gadouas/Desktop/Projet AFM/Data/Preprocessed_data//patient 20/tumeur/map 4_processed-2019.09.20-13.31.36.xlsx",                                            
# "/Users/gauthier.gadouas/Desktop/Projet AFM/Data/Preprocessed_data//patient 3/tissu_sain/epithelium/map 1vertical-tip-position_processed-2018.06.22-13.38.00.xlsx",         
# "/Users/gauthier.gadouas/Desktop/Projet AFM/Data/Preprocessed_data//patient 3/tissu_sain/muscle/map 1 vertical-tip-position_processed-2018.06.22-13.48.52.xlsx",            
# "/Users/gauthier.gadouas/Desktop/Projet AFM/Data/Preprocessed_data//patient 3/tumeur/epithelium/manip 2/map 1 vertical-tip-position_processed-2018.06.22-13.24.32.xlsx",    
# "/Users/gauthier.gadouas/Desktop/Projet AFM/Data/Preprocessed_data//patient 4/tissu_sain/epithelium/manip 1/map 1 vertical-tip-position_processed-2018.06.22-14.04.19.xlsx",
# "/Users/gauthier.gadouas/Desktop/Projet AFM/Data/Preprocessed_data//patient 5/tissu_sain/epithelium/manip 1/map 2 N5OCT 1_processed-2018.09.28-12.59.11.xlsx"]

#path = "/Users/gauthier.gadouas/Desktop/Projet AFM/Data/Preprocessed_data"

def remove_outliers(df, column_name, threshold=2):
    # Calculate the median
    median_val = df[column_name].median()

    # Calculate the MAD (Median Absolute Deviation)
    mad = np.median(np.abs(df[column_name] - median_val))

    # Calculate the modified Z-score
    df['modified_z_score'] = 0.6745 * (df[column_name] - median_val) / mad

    # Identify and remove outliers based on the threshold
    outliers = df[np.abs(df['modified_z_score']) > threshold]
    df_no_outliers = df[np.abs(df['modified_z_score']) <= threshold]

    # Remove the temporary 'modified_z_score' column
    df_no_outliers = df_no_outliers.drop(columns=['modified_z_score'])

    return df_no_outliers, outliers


# Function to calculate weighted distances
def calculate_weighted_distances(df, threshold):
    # Standardize the weighted distances
    scaler = StandardScaler()
    df["Data"] = scaler.fit_transform(df["Data"].to_numpy().reshape(-1, 1))
    print(df["Data"])

    # Calculate Euclidean distances between all pairs of points
    distances = squareform(pdist(df[['X Position', 'Y Position']].values))
    
    # Calculate weights based on distances
    weights = np.where(distances < 6E-6, 1, np.where((distances >= 6E-6) & (distances < 11E-6), 0.5, np.where(distances >= threshold, 0, np.nan)))
    
    # Calculate weighted distances
    weighted_distances = np.nansum(weights * np.abs(df["Data"].values[:, None] - df["Data"].values[None, :]), axis=1)

    # Count the number of points where the weights are not equal to 0
    count_nonzero_weights = np.count_nonzero(weights, axis=1)
    
    # Divide the weighted distances by the number of points where the weights are not equal to 0
    weighted_distances /= 2*count_nonzero_weights

    # Divide the weighted distances by the mean of the "Young's Modulus [Pa]" column
    #weighted_distances /= df["Young's Modulus [Pa]"].mean()
    
    return weighted_distances


path = "/Users/gauthier.gadouas/Desktop/Projet AFM/Data/Original_data_threshold"

#path = "/Users/gauthier.gadouas/Desktop/Projet AFM/Data/Original_data"

clinical_data = pd.ExcelFile("/Users/gauthier.gadouas/Desktop/Projet AFM/Data/Listing_patients_nanomecaniques_ICM_2020218_Anonymise.xlsx").parse()

#Ensure the data are converted to datetime in each column with datetime
columns_to_convert = [ clinical_data["DDN"], clinical_data["Date de diagnostic"] ]
for column in columns_to_convert:
    for k in range(len(column)):
        if isinstance(column[k], datetime.date) != True:
            split = column[k].split("/")
            column[k] = datetime.date(int(split[-1]), int(split[-2]), int(split[-3]))      

# Recursively walk into the path to get files ending by .xlsx
exl_in_path = list()
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith('.xlsx'):
           exl_in_path.append(os.path.join(root, file))
exl_in_path = [item for item in exl_in_path if 'patient ' in item]

#Remove maps
exl_in_path = list(set(exl_in_path)) #- set(removed_maps))

# Iteration over the cohort of 20 patient
N = 20
list_df_concatenate = list()
list_df_concatenate_2 = list()
for k in range(0, N):
    # create a sublist of exl per patient
    #sublist_exl = [item for item in exl_in_path if f'patient {k}' in item]
    sublist_exl = [item for item in exl_in_path if float(item.split("/")[7].split(" ")[-1]) == float(k+1)]
    list_of_types = list()
    for path in sublist_exl:
        df = pd.ExcelFile(path).parse()
        Kek = path
        #data = remove_outliers(df, "Young's Modulus [Pa]", threshold = 2)
        # Group by 'X Position' and 'Y Position' and calculate the mean of "Young's Modulus [Pa]"
        grouped = df.groupby(['X Position', 'Y Position'])["Young's Modulus [Pa]"].mean().reset_index()
        # Replace df with the grouped DataFrame
        df = grouped
        df = df.dropna(axis=0).reset_index(drop=True)

        df['X Position'] = pd.to_numeric(df['X Position'], errors='coerce')
        df['Y Position'] = pd.to_numeric(df['Y Position'], errors='coerce')
        df["Data"] = pd.to_numeric(df["Young's Modulus [Pa]"], errors='coerce')
        data = df["Data"].rename("Data")

        df['WeightedDistances'] = calculate_weighted_distances(df, threshold = 11E-6)

        log_data = data.apply(lambda x: np.log10(x + 1)).rename("Log Data")
        
        #Extract intel on clinical data for patient k
        clinical_dfk = pd.concat([clinical_data[clinical_data["Numéro Patient AFM"] == k+1]]*len(data), ignore_index= True)

        #Extract also type of sample 
        subpath = re.search(f"patient {k+1}(.*).xlsx", path).group().split("/")[1:]
        #typ = [subpath[0] + subpath[1]]
        map_word = {"metastase" : "Metastase",
                    "tissu sain" : "Sain",
                    "tissu sain proxi" : "Proxi",
                    "tumeur" : "Tumeur",
                    "epithelium" : "Epith",
                    "muscle" : "Muscle",
                    "matrice": "Matrice",
                    "stroma": "Stroma"
                    }
        if any("manip" in sub for sub in subpath[:-1]):
            a = subpath[:-2]
            typ = str()
            for j in range(len(a)):
                typ += a[j]+" "
        else:
            a = subpath[:-1]
            typ = str()
            for j in range(len(a)):
                typ += a[j]+" "
        if typ not in list_of_types:
            list_of_types.append(typ)
            M = 1
        else:
            M += 1
            
        labelling = {"tumeur " : "Tu",
                     "tumeur stroma " : "TuS",
                     "tumeur epithelium ": "TuE",
                     "tissu_sain epithelium " : "SE",
                     "tissu_sain muscle ": "SMu",
                     "tissu_sain matrice ": "SMa",
                     "tissu_sain stroma ": "SS",
                     "tissu_sain_proxi ": "Prox",
                     "tissu_sain_proxi epithelium ": "ProxE",
                     "metastase ": "Meta",
                     "metastase stroma ": "MetaS",
                     "metastase epithelium ": "MetaE",
            }
        
        summary = f"P{k+1}-"+labelling[typ]+f"-M{M}"
        new_df = pd.DataFrame({"Tissu" : [typ], "Map" : [summary] } )#[typ.split(" ")[-2]]
        new_df = pd.concat([new_df]*len(data), ignore_index= True)
        
        df_final = pd.concat([data, log_data, new_df, clinical_dfk], axis = 1)
        list_df_concatenate.append(df_final)

        #Weighted distances calculus
        df_final_2 = pd.concat([data, new_df, df['X Position'], df['Y Position'], df['WeightedDistances']], axis = 1)
        list_df_concatenate_2.append(df_final_2)
        
df_final = list_df_concatenate[0]        
for k in range (1, len(list_df_concatenate)):
    df_final = pd.concat([df_final, list_df_concatenate[k]], axis = 0, ignore_index = True)
df_final=df_final.dropna(axis=0, subset=["Data"]).reset_index(drop=True).infer_objects()
df_final.to_excel('all_data_threshold.xlsx', index = False)

df_final_2 = list_df_concatenate_2[0]        
for k in range (1, len(list_df_concatenate_2)):
    df_final_2 = pd.concat([df_final_2, list_df_concatenate_2[k]], axis = 0, ignore_index = True)
df_final_2=df_final_2.dropna(axis=0, subset=["Data"]).reset_index(drop=True).infer_objects()
df_final_2.to_excel('all_data_distances.xlsx', index = False)


f, boxplot = plt.subplots(figsize=(3*len(df_final["Tissu"].unique()), 6))
palette = sns.color_palette("CMRmap")
boxplot = sns.set_style("white")
# boxplot set up and box-whis style
boxplot = sns.boxplot(palette=palette, 
                      data=df_final, x= "Tissu", y="Log Data",# hue="Tissu", 
                      dodge=False, orient="v", saturation = 0.8,
                      boxprops = dict(linewidth=1.5, edgecolor='black', alpha = 0.8),
                      whiskerprops = dict(linewidth=1.5, color='black'),
                      capprops = dict(linewidth=1.5, color='black'),
                      flierprops=dict(marker="d", markerfacecolor= "black", markeredgecolor="black", 
                                      markersize =0.75, alpha=0.2),
                      medianprops=dict(color="black", linewidth=1.5, linestyle= '--'), 
                      showmeans=True,
                      meanprops=dict(marker="o", markerfacecolor="black", alpha=1.0,
                                     markeredgecolor="black", markersize=6, linewidth=0.2, zorder=10))

boxplot = sns.stripplot(data=df_final ,x= "Tissu", y="Log Data", marker="o", edgecolor='white', 
                        alpha=0.3, size=1.5, linewidth=0.6, color='black', jitter = True, zorder=0)
boxplot.set_title("All patients", fontsize=18, weight = "bold")
boxplot.set_ylabel("Log Young Modulus (Pa)", fontsize=14, color='black', weight="bold")
boxplot.set_xlabel("Tissu", fontsize=14, color='black', weight="bold")
#boxplot.set(ylim=(0.48, 1.02), yticks=np.arange(0.5,1.02,0.05))
#sns.despine(left=False, bottom=False)
# x-axis rotation and text color
boxplot.set_xticklabels(boxplot.get_xticklabels(),rotation = 0, color='black', fontsize=20)
# x-axis and y-axis tick color
boxplot.tick_params(colors='black', which='both')  # 'both' refers to minor and major axes
# x-axis and y-axis label color
boxplot.axes.yaxis.label.set_color('black')
boxplot.axes.xaxis.label.set_color('black')
# format graph outline (color)
boxplot.spines['left'].set_color('black')
boxplot.spines['bottom'].set_color('black')
boxplot.spines['right'].set_color('black')
boxplot.spines['top'].set_color('black')
# add tick marks on x-axis or y-axis
boxplot.tick_params(bottom=False, left=True)
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
plt.tight_layout()


##### Get boxplots for each patient with respect to the analyzed tissues ########
grouped_data = df_final.groupby("Numéro Patient AFM")
for data_per_patient in grouped_data:
    f, boxplot = plt.subplots(figsize=(3*len(data_per_patient[1]["Tissu"].unique()), 6))
    palette = sns.color_palette("CMRmap")
    boxplot = sns.set_style("white")
    # boxplot set up and box-whis style
    boxplot = sns.boxplot(palette=palette, 
                          data=data_per_patient[1], x= "Tissu", y="Log Data",# hue="Tissu", 
                          dodge=False, orient="v", saturation = 0.8,
                          boxprops = dict(linewidth=1.5, edgecolor='black', alpha = 0.8),
                          whiskerprops = dict(linewidth=1.5, color='black'),
                          capprops = dict(linewidth=1.5, color='black'),
                          flierprops=dict(marker="d", markerfacecolor= "black", markeredgecolor="black", 
                                          markersize =0.75, alpha=0.2),
                          medianprops=dict(color="black", linewidth=1.5, linestyle= '--'), 
                          showmeans=True,
                          meanprops=dict(marker="o", markerfacecolor="black", alpha=1.0,
                                         markeredgecolor="black", markersize=6, linewidth=0.2, zorder=10))

    boxplot = sns.stripplot(data=data_per_patient[1] ,x= "Tissu", y="Log Data", marker="o", edgecolor='white', 
                            alpha=0.3, size=1.5, linewidth=0.6, color='black', jitter = True, zorder=0)
    boxplot.set_title("Patient "+str(data_per_patient[0]), fontsize=18, weight = "bold")
    boxplot.set_ylabel("Log Young Modulus (Pa)", fontsize=14, color='black', weight="bold")
    boxplot.set_xlabel("Tissu", fontsize=14, color='black', weight="bold")
    #boxplot.set(ylim=(0.48, 1.02), yticks=np.arange(0.5,1.02,0.05))
    #sns.despine(left=False, bottom=False)
    # x-axis rotation and text color
    boxplot.set_xticklabels(boxplot.get_xticklabels(),rotation = 0, color='black', fontsize=20)
    # x-axis and y-axis tick color
    boxplot.tick_params(colors='black', which='both')  # 'both' refers to minor and major axes
    # x-axis and y-axis label color
    boxplot.axes.yaxis.label.set_color('black')
    boxplot.axes.xaxis.label.set_color('black')
    # format graph outline (color)
    boxplot.spines['left'].set_color('black')
    boxplot.spines['bottom'].set_color('black')
    boxplot.spines['right'].set_color('black')
    boxplot.spines['top'].set_color('black')
    # add tick marks on x-axis or y-axis
    boxplot.tick_params(bottom=False, left=True)
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.tight_layout()


###### Plot an histogram for the density and so on

    



