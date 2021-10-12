#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import variance
#merge three table
data_order_via_price = pd.read_csv('order_via_price.csv')
data_meal_info = pd.read_csv('meal_info.csv')
data_fulfilment_center_info = pd.read_csv('fulfilment_center_info.csv')
data_info = pd.DataFrame(columns= ['name_data', 'size_data', 'columns_data'])
data_info.loc[0] = ['order_via_price',data_order_via_price.shape,[data_order_via_price.columns]]
data_info.loc[1] = ['meal_info',data_meal_info.shape,[data_meal_info.columns]]
data_info.loc[2] = ['fulfilment_center',data_fulfilment_center_info.shape,[data_fulfilment_center_info.columns]]
pd.set_option('display.max_colwidth', None)
#print(data_order_via_price.shape)
#data_info.head()
data = data_order_via_price
data = data.merge(data_meal_info, left_on = 'meal_id', right_on = 'meal_id')
data = data.merge(data_fulfilment_center_info , left_on = 'center_id', right_on = 'center_id')
#print(data.shape)
#data.columns
# # ------ EDA ------
#clean data
missing_data = data.isnull().sum(axis=0).reset_index()
missing_data.columns = ['variable', 'missing values']
missing_data['filling factor (%)']=(data.shape[0]-missing_data['missing values'])/data.shape[0]*100
missing_data.sort_values('filling factor (%)').reset_index(drop = True)

def pie_plot(data, feature1, feature2, p1, explo):
    #dictionary for feature11 and its total feature2
    d_feature1 = {}
    
    #total feature2
    total = data[feature2].sum()

    #find ratio of feature2 per feature1
    for i in range(data[feature1].nunique()):

        #cuisine
        c = data[feature1].unique()[i]

        #num of feature2 for feature1
        c_feature2 = data[data[feature1]==c][feature2].sum()
        d_feature1[c] = c_feature2/total

     #pie plot 
    p1.pie([x*100 for x in d_feature1.values()],labels=[x for x in d_feature1.keys()],autopct='%0.1f',explode = explo) 

     #label the plot 
    p1.set_title('{} via {}'.format(feature2, feature1)) 
    
################################################################################################
#plotting histogram 
def hist_plot(data, feature1, feature2, theta, space):
    plt.rcParams.update({'font.size': 15})
    fig, axs = plt.subplots(1,2, figsize = (16,5+2*space))
    plt.subplots_adjust(bottom=space)
    axs[0].hist(data[feature1],rwidth=0.9,alpha=0.3,color='blue',bins=25,edgecolor='red')
    axs[1].hist(data[feature2],rwidth=0.9,alpha=0.3,color='blue',bins=25,edgecolor='red')

    #x and y-axis labels 
    axs[0].set(xlabel=feature1) 
    axs[1].set(xlabel=feature2) 
    axs[0].set(ylabel='Frequency') 
    plt.xticks(rotation=theta)
    #plot title 
    #axs[0].set(title='Inspecting {} effect'.format(feature1)) 
    #axs[1].set(title='Inspecting {} effect'.format(feature2))
    plt.savefig('images/hist_plot_{}_and_{}.png'.format(feature1, feature2))
    plt.show()

###################################################################################################
def scatter_plot(data,feature1, feature2, feature_as_color, list_feature_as_color,plt):
    plt.figure(figsize=(10,5))
    scatter = plt.scatter(data[feature1],
            data[feature2], s=10,
            c=data[feature_as_color].astype('category').cat.codes, alpha = .4)
    plt.xlabel(feature1, size=15)
    plt.ylabel(feature2, size=15)
    # add legend to the plot with names
    plt.legend(handles=scatter.legend_elements()[0], 
           labels=list_feature_as_color,
           title=feature_as_color)
    plt.savefig('images/scatter_plot_{}_via_{}_{}.png'.format(feature1, feature2,feature_as_color))
    plt.show()
#-------------------------------------------------------------------------------------
def scatter_log_plot(data1, featur1, featur2, featur3, names,plt):
    dt = data1
    dt['log_{}'.format(featur1)] = np.log(data.num_orders)
    dt['{}_per_{}'.format(featur2, featur1)] = data[featur2]/data[featur1]
    scatter_plot(dt, 'log_{}'.format(featur1), '{}_per_{}'.format(featur2, featur1), featur3, names,plt)
###################################################################################################
def bar_plot(data, feature1, feature2, feature3,plt, space):
    plt.figure(figsize=(7,5))
    plt.subplots_adjust(bottom=space)
    plt.rcParams.update({'font.size': 15})
    sns.barplot(x=feature1, y=feature2, data=data, color = 'red', label = feature2, alpha = 0.4)
    sns.barplot(x=feature1, y=feature3, data=data, color = 'blue', label = feature3, alpha = 0.4)
    plt.xticks(rotation=90)
    plt.xlabel(feature1, size=15)
    plt.ylabel("")
    plt.legend()
    plt.savefig('images/bar_plot_{}_via_{}_and_{}.png'.format(feature1, feature2, feature3))
    plt.show()
#####################################################################################################
def plot_mean_std(data,feature1, feature2):
    #var = sqrt(sum(i-mean)**2/N)
    dt = data[[feature1, feature2]].groupby([feature1])
    plt.figure(figsize=(10,5))
    plt.rcParams.update({'font.size': 15})
    plt.scatter(data[feature1].unique(),dt.mean(), marker="o", label= 'mean')
    plt.scatter(data[feature1].unique(),dt.std(), marker="*", label = 'std')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend()
    plt.savefig('images/mean_std_plot_{}_via_{}.png'.format(feature1, feature2))
    plt.show()
######################################################################################################
def plot_total_value_features(data,feature1, feature2):
    dt = data[[feature1, feature2]].groupby([feature1])
    plt.figure(figsize=(10,5))
    plt.rcParams.update({'font.size': 15})
    plt.scatter(data[feature1].unique(),dt.sum(), marker="o")
    plt.xlabel(feature1)
    plt.ylabel('sum_over_{}'.format(feature2))
    plt.legend()
    plt.savefig('images/sum_over_features_plot_{}_via_{}.png'.format(feature1, feature2))
    plt.show()

# # ________pie_plot______
plt.rcParams.update({'font.size': 15})
fig, axs = plt.subplots(1,2, figsize = (14,5))
pie_plot(data, 'emailer_for_promotion', 'num_orders', axs[0], [0,0])
pie_plot(data, 'emailer_for_promotion', 'checkout_price', axs[1], [0,0])
plt.savefig('images/pie_plot_{}_via_{}_and_{}.png'.format('emailer_for_promotion', 'num_orders','checkout_price'))
plt.show()
#----------------------------------------------------------------
plt.rcParams.update({'font.size': 15})
fig, axs = plt.subplots(1,2, figsize = (14,5))
pie_plot(data, 'homepage_featured', 'num_orders', axs[0], [0,0])
pie_plot(data, 'homepage_featured', 'checkout_price', axs[1], [0,0])
plt.savefig('images/pie_plot_{}_via_{}_and_{}.png'.format('homepage_featured', 'num_orders','checkout_price'))
plt.show()
#----------------------------------------------------------------
plt.rcParams.update({'font.size': 15})
fig, axs = plt.subplots(1,2, figsize = (14,5))
pie_plot(data, 'category', 'num_orders', axs[0], [0,0.2,0.0,0.2,0.,0.2,0.0,0.2,0.0,0.4,0.0,.2,0.0,.2])
pie_plot(data, 'category', 'checkout_price', axs[1], [0,0.1,0.0,0.1,0.,0.2,0.0,0.2,0.0,0.2,0.0,.2,0.0,.2])
plt.savefig('images/pie_plot_{}_via_{}_and_{}.png'.format('category', 'num_orders','checkout_price'))
plt.show()
#----------------------------------------------------------------
plt.rcParams.update({'font.size': 15})
fig, axs = plt.subplots(1,2, figsize = (14,5))
pie_plot(data, 'cuisine', 'num_orders', axs[0], [0,0,0,0])
pie_plot(data, 'cuisine', 'checkout_price', axs[1], [0,0,0,0])
plt.savefig('images/pie_plot_{}_via_{}_and_{}.png'.format('cuisine', 'num_orders','checkout_price'))
#----------------------------------------------------------------
plt.rcParams.update({'font.size': 15})
fig, axs = plt.subplots(1,2, figsize = (14,5))
pie_plot(data, 'region_code', 'num_orders', axs[0], [0,0.0,0,0,0.4,0.8,1.2,1.5])
pie_plot(data, 'region_code', 'checkout_price', axs[1], [0,0.0,0,0,0.4,0.8,1.2,1.5])
plt.savefig('images/pie_plot_{}_via_{}_and_{}.png'.format('region_code', 'num_orders','checkout_price'))
plt.show()
#----------------------------------------------------------------
plt.rcParams.update({'font.size': 15})
fig, axs = plt.subplots(1,2, figsize = (14,5))
pie_plot(data, 'center_type', 'num_orders', axs[0], [0,0.0,0])
pie_plot(data, 'center_type', 'checkout_price', axs[1], [0,0.0,0])
plt.savefig('images/pie_plot_{}_via_{}_and_{}.png'.format('center_type', 'num_orders','checkout_price'))
plt.show()
#----------------------------------------------------------------

# # ______hist_plot______
hist_plot(data, 'checkout_price', 'num_orders',0, .2)
hist_plot(data, 'week', 'center_id', 0, .2)
hist_plot(data, 'meal_id', 'base_price', 0, .2)
hist_plot(data, 'emailer_for_promotion', 'homepage_featured', 0, 0.2)
hist_plot(data,  'cuisine', 'category', 60, 0.36)
hist_plot(data, 'city_code', 'region_code', 0, .2)
hist_plot(data, 'center_type', 'op_area', 0, .2)

# # ______scatter_plot_____
cuisine_names = ['Thai', 'Indian', 'Italian', 'Continental']
category_names = ['Beverages', 'Rice Bowl', 'Starters', 'Pasta', 'Sandwich',
                  'Biryani', 'Extras', 'Pizza', 'Seafood', 'Other Snacks', 
                  'Desert', 'Salad', 'Fish', 'Soup']
homepage_featured_names = [1,0]
emailer_for_promotion_names = [1,0]
region_code_names = [56,85,77,34,35,71,93,23]
center_type_names = ['TYPE_A', 'TYPE_B', 'TYPE_C']
op_area_names = [4.0,3.9,3.8,4.4,4.5,2.8,4.1,7.0,4.8,3.4,3.6,5.1,4.2,2.7,3.0,2.0,6.7,6.3,5.6,3.7,3.5,3.2,5.0,5.3,4.6, 4.7,2.4,2.9,1.9,0.9]
scatter_plot(data, 'checkout_price', 'num_orders', 'cuisine', cuisine_names,plt)
scatter_log_plot(data, 'num_orders', 'checkout_price', 'cuisine', cuisine_names,plt)
#---------------------------------------------------------------------
scatter_plot(data, 'checkout_price', 'num_orders', 'homepage_featured', homepage_featured_names,plt)
scatter_log_plot(data, 'num_orders', 'checkout_price', 'homepage_featured', homepage_featured_names,plt)
#----------------------------------------------------------------------------------
scatter_plot(data, 'checkout_price', 'num_orders', 'emailer_for_promotion', emailer_for_promotion_names,plt)
scatter_log_plot(data, 'num_orders', 'checkout_price', 'emailer_for_promotion', emailer_for_promotion_names,plt)
#----------------------------------------------------------------------------------
scatter_plot(data, 'checkout_price', 'num_orders', 'region_code', region_code_names,plt)
scatter_log_plot(data, 'num_orders', 'checkout_price', 'region_code', region_code_names, plt)
#----------------------------------------------------------------------------------
scatter_plot(data, 'checkout_price', 'num_orders', 'center_type', center_type_names,plt)
scatter_log_plot(data, 'num_orders', 'checkout_price', 'center_type', center_type_names, plt)
#----------------------------------------------------------------------------------
scatter_plot(data, 'checkout_price', 'num_orders', 'op_area',op_area_names,plt)
scatter_log_plot(data, 'num_orders', 'checkout_price', 'op_area', op_area_names,plt)

# #  ______bar_plot_____
bar_plot(data, 'emailer_for_promotion', 'checkout_price', 'num_orders',plt,.1)
bar_plot(data, 'homepage_featured', 'checkout_price', 'num_orders',plt,.1)
bar_plot(data, 'category', 'checkout_price', 'num_orders',plt,0.36)
bar_plot(data, 'cuisine', 'checkout_price', 'num_orders',plt,0.31)
bar_plot(data, 'region_code', 'checkout_price', 'num_orders',plt,.12)
bar_plot(data, 'center_type', 'checkout_price', 'num_orders',plt,.22)
bar_plot(data, 'op_area', 'checkout_price', 'num_orders',plt,.15)

# #  _______ mean_std_plot______
plot_mean_std(data, 'week', 'checkout_price')
plot_mean_std(data, 'week', 'emailer_for_promotion')
plot_mean_std(data, 'week', 'homepage_featured')
plot_mean_std(data, 'week', 'num_orders')

# # __________ sum_over_features_________
plot_total_value_features(data, 'week', 'checkout_price')
plot_total_value_features(data, 'week', 'emailer_for_promotion')
plot_total_value_features(data, 'week', 'homepage_featured')
plot_total_value_features(data, 'week', 'num_orders')
#---------------------------------------------------------------
data['price_per_order'] = data['checkout_price']/data['num_orders']
plot_total_value_features(data, 'week', 'price_per_order')

# # ----- A/B Testing-----
#function to calculte P_value
from scipy.stats import ttest_ind
from scipy import stats
from statsmodels.stats import weightstats as stests
from scipy.stats import mannwhitneyu
import numpy as np
def cal_p_val(dt, feature_1, feature_2, cuisine1, cuisine2):
    result_ttest,result_utest = 'reject Null', 'reject Null'
    data_A = dt[dt[feature_1] == cuisine1]
    data_B = dt[dt[feature_1] == cuisine2]
    if len(data_A) !=0 and len(data_B) != 0:
                ttest,pval_t_0 = ttest_ind(data_A[feature_2],data_B[feature_2])
                stat, pval_mann_0 = mannwhitneyu(data_A[feature_2],data_B[feature_2])
                if pval_t_0 > alpha: result_ttest = 'not reject Null'
                if pval_mann_0 > alpha: result_utest = 'not reject Null'
    return pval_t_0, pval_mann_0, round(data_A[feature_2].mean(),2), round(data_B[feature_2].mean(),2), len(data_A), len(data_B),result_ttest, result_utest
#-------------------------------------------------------------------------------
def table_info(dt,feature_1, feature_2, list_feature):
    table_info = pd.DataFrame(columns= ['Sample A', 'Sample B', 'pval_ttest', 'pval_utest', 'mean_{}_A'.format(feature_2), 'mean_{}_B'.format(feature_2), 'n_A', 'n_B', 't-test', 'u-test'])
    k = 0
    alpha = 0.05
    for i in range(len(list_feature)):
        for j in range(i+1, len(list_feature)):
            a, b, c, d, e, f, g, h = cal_p_val(dt, feature_1, feature_2, list_feature[i], list_feature[j])
            table_info.loc[k] = list_feature[i], list_feature[j], a, b, c, d, e, f, g ,h
            k += 1
    import subprocess
    table_info.to_html('table2.html')
    subprocess.call('wkhtmltoimage -f png --width 0 table2.html images/table_{}_{}.png'.format(feature_1, feature_2), shell=True)
    return table_info
#--------------------------------------------------------------------------------
def table_info_via_threshold(dt,feature_1, feature_2,a,b,c):
    table_info = pd.DataFrame(columns= ['Threshold','Sample A', 'Sample B', 'pval_ttest', 'pval_utest', 'mean_{}_A'.format(feature_2), 'mean_{}_B'.format(feature_2), 'n_A', 'n_B', 't-test', 'u-test'])
    k = 0
    alpha = 0.05
    for p in range(a,b,c):
        def f(x):
            if x <= p:
                return 'before'
            else:
                return 'after'
        dt['week_label'] = dt[feature_1].apply(f)
        list_feature = ['before', 'after']
        for i in range(len(list_feature)):
            for j in range(i+1, len(list_feature)):
                a, b, c, d, e, f, g, h = cal_p_val(dt, 'week_label', feature_2, list_feature[i], list_feature[j])
                table_info.loc[k] = p,list_feature[i], list_feature[j], a, b, c, d, e, f, g ,h
                k += 1
        dt.drop(['week_label'], axis=1, inplace= True)        
    import subprocess
    table_info.to_html('table2.html')
    subprocess.call('wkhtmltoimage -f png --width 0 table2.html images/table_{}_{}_via_treshold.png'.format(feature_1, feature_2), shell=True)
    return table_info
#------------------------------------------------------------------------------------
def table_info_via_three_parts(dt,feature_1, feature_2, list_feature):
    table_info = pd.DataFrame(columns= ['Sample A', 'Sample B', 'pval_ttest', 'pval_utest', 'mean_{}_A'.format(feature_2), 'mean_{}_B'.format(feature_2), 'n_A', 'n_B', 't-test', 'u-test'])
    k = 0
    alpha = 0.05
    for i in range(len(list_feature)-1):
            j =i+1
            a, b, c, d, e, f, g, h = cal_p_val(dt, feature_1, feature_2, list_feature[i], list_feature[j])
            table_info.loc[k] = list_feature[i], list_feature[j], a, b, c, d, e, f, g ,h
            k += 1
    import subprocess
    table_info.to_html('table2.html')
    subprocess.call('wkhtmltoimage -f png --width 0 table2.html images/table_{}_{}_three_parts_week.png'.format(feature_1, feature_2), shell=True)
    return table_info

# # ----- A/B Testing based on cuisions-----
# significance level
alpha = 0.05
# Null hypothesis 
### checkout price of Italian cuisine is the same as the chekckout price Thai cuisine!
# alternative 
### checkout price of Italian and Thia cuisines is not the same!
#if pval <0.05:
#  print("we reject null hypothesis")
#else:
#  print("we can not reject null hypothesis")
table_info(data, 'cuisine', 'checkout_price', cuisine_names)
table_info(data, 'cuisine', 'num_orders', cuisine_names)

# # ----- A/B Testing based on categories-----
table_info(data, 'category', 'checkout_price', category_names)
table_info(data, 'category', 'num_orders', category_names)

# # ----- A/B Testing based on homepage_featured-----
# significance level
alpha = 0.05
# Null hypothesis 
### checkout price with homepage_featured=0 is the same as the chekckout price with homepage_featured=1!
# alternative 
### checkout price is not the same for both!
#if pval <0.05:
#  print("we reject null hypothesis")
#else:
#  print("we can not reject null hypothesis")
table_info(data, 'homepage_featured', 'checkout_price', homepage_featured_names)
table_info(data, 'homepage_featured', 'num_orders', homepage_featured_names)

# # ----- A/B Testing based on emailer_for_promotion-----
table_info(data, 'emailer_for_promotion', 'checkout_price', emailer_for_promotion_names)
table_info(data, 'emailer_for_promotion', 'num_orders', emailer_for_promotion_names)

# # ----- A/B Testing based on center_type-----
table_info(data, 'center_type', 'checkout_price', center_type_names)
table_info(data, 'center_type', 'num_orders', center_type_names)

# # ----- A/B Testing based on week-----
table_info_via_threshold(data, 'week', 'checkout_price', 10, 145,30)
table_info_via_threshold(data, 'week', 'num_orders', 10, 145,30)

def f(x):
    if x<50:
        return '0-49'
    if 50 <= x <100:
        return '50-99'
    if x>=100:
        return '100-145'
data['week_label'] = data['week'].apply(f)
week_label_names = ['0-49','50-99','100-145']
table_info_via_three_parts(data, 'week_label', 'num_orders', week_label_names)
def f(x):
    if x<50:
        return '0-49'
    if 50 <= x <100:
        return '50-99'
    if x>=100:
        return '100-145'
data['week_label'] = data['week'].apply(f)
week_label_names = ['0-49','50-99','100-145']
table_info_via_three_parts(data, 'week_label', 'checkout_price', week_label_names)

