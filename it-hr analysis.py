
import pandas as pd
import numpy as np
np.float = float    
np.int = int   #module 'numpy' has no attribute 'int'
np.object = object    #module 'numpy' has no attribute 'object'
np.bool = bool    #module 'numpy' has no attribute 'bool'
import matplotlib
import matplotlib.pyplot as plt
import sweetviz as sv
import seaborn as sns
import scipy.stats as stats
from scipy.stats import kendalltau
from numpy.random import rand
from numpy.random import seed
from scipy.stats import spearmanr
from scipy.stats import kruskal

def cronbach_alpha(data):
    # Transform the data frame into a correlation matrix
    df_corr = data.corr()
    
    # Calculate N
    # The number of variables is equal to the number of columns in the dataframe
    N = data.shape[1]
    
    # Calculate r
    # For this, we'll loop through all the columns and append every
    # relevant correlation to an array called 'r_s'. Then, we'll
    # calculate the mean of 'r_s'.
    rs = np.array([])
    for i, col in enumerate(df_corr.columns):
        sum_ = df_corr[col][i+1:].values
        rs = np.append(sum_, rs)
    mean_r = np.mean(rs)
    
   # Use the formula to calculate Cronbach's Alpha 
    cronbach_alpha = (N * mean_r) / (1 + (N - 1) * mean_r)
    return cronbach_alpha
	
	
## Automated analytics

#produce the analytical report
df = pd.read_excel('it_dataset.xlsx')
#analyzing the dataset
report = sv.analyze(df)
#display the report
report.show_html('automated_report.html')

## Customized EDA

df = pd.read_excel('it_dataset_recoded.xlsx')

df.describe()

fig = plt.figure(figsize = (15,100))

sns.set(font_scale=2)
fig.subplots_adjust(hspace = 0.8)

ax1 = fig.add_subplot(17,1,1)
sns.countplot(data = df, x = 'satisfaction', ax=ax1).set(title='Assess the overall satisfaction from your current role')

ax2 = fig.add_subplot(17,1,2)
sns.countplot(data = df, x = 'searching_new_job', ax=ax2).set(title='How likely is to search for another position in the near future (6 months)?')

ax3 = fig.add_subplot(17,1,3)
sns.countplot(data = df, x = 'compensation_satisfaction', ax=ax3).set(title='Do you feel satisfied with the compensation package (including salary, bonus, merit, etc) that you receive?')

ax4 = fig.add_subplot(17,1,4)
sns.countplot(data = df, x = 'motivation', ax=ax4).set(title='Do you feel inspired/motivated at your current role?')

ax5 = fig.add_subplot(17,1,5)
sns.countplot(data = df, x = 'growth', ax=ax5).set(title='Do you think that you will have the chance to grow in this organization based on objective criteria ?')

ax6 = fig.add_subplot(17,1,6)
sns.countplot(data = df, x = 'fairness', ax=ax6).set(title='Assess the level of fairness and transparency in your organization.')


ax7 = fig.add_subplot(17,1,7)
sns.countplot(data = df, x = 'flexibility', ax=ax7).set(title='Assess the level of flexibility that you have in your current role (i.e., remote working options)')

ax8 = fig.add_subplot(17,1,8)
sns.countplot(data = df, x = 'full_capacity', ax=ax8).set(title='Do you think that you operate at your full capacity at this role?')

ax9 = fig.add_subplot(17,1,9)
sns.countplot(data = df, x = 'overtime_working_freq', ax=ax9).set(title='Do you usually work overtime?')

ax10 = fig.add_subplot(17,1,10)
sns.countplot(data = df, x = 'learning', ax=ax10).set(title='Does your employer provide learning opportunities?')

ax11 = fig.add_subplot(17,1,11)
sns.countplot(data = df, x = 'discrepancy_promised_received', ax=ax11).set(title='Do you think that there is a discrepancy between what you were promised from your employer compared to what you finally received?')

ax12 = fig.add_subplot(17,1,12)
sns.countplot(data = df, x = 'sense_of_belonging', ax=ax12).set(title='Do you feel the sense of belonging in your company?')


ax13 = fig.add_subplot(17,1,13)
sns.countplot(data = df, x = 'values_alignement', ax=ax13).set(title="Do you feel that your personal values are in accordance with the company's purpose/mission?")

ax14 = fig.add_subplot(17,1,14)
sns.countplot(data = df, x = 'working_env', ax=ax14).set(title='Do you enjoy the working environment/conditions (i.e., ergonomic, accessible);')

ax15 = fig.add_subplot(17,1,15)
sns.countplot(data = df, x = 'supervisors_support', ax=ax15).set(title='Do you receive guidance/support from your supervisor?')

ax16 = fig.add_subplot(17,1,16)
sns.countplot(data = df, x = 'joy', ax=ax16).set(title="Do you enjoy working at this organization (i.e., company’s culture, interaction with colleagues, fun activities)?")

ax17 = fig.add_subplot(17,1,17)
sns.countplot(data = df, x = 'security', ax=ax17).set(title='How much security/stability does your organization provides you?')

features = df.columns.delete(0)

## Chi-square test

cat_variable_list =[]
chi2_list =[]
p_list =[]
dof_list = []
message_list =[]

for cat_variable in features:      
        table = pd.crosstab(df['satisfaction'], df[cat_variable]==1)
        chi2, p, dof, ex = stats.chi2_contingency(table)
     
        # calculate kendall's correlation
        
        # interpret the significance
        alpha = 0.05
        if p > alpha:
            message ='Samples are uncorrelated (fail to reject H0)'
        else:
            message='Samples are correlated (reject H0)'
        
        cat_variable_list.append(cat_variable)
        chi2_list.append(round(chi2,3))
        p_list.append(round(p,3))
        dof_list.append(dof)
        message_list.append(message)

out= pd.DataFrame({'variable': cat_variable_list, 'p':p_list, 'chi2': chi2_list, 'dof': dof_list, 'comment':message_list })      

## Spearman's rank

cat_variable_list =[]
coef_list =[]
p_list =[]
message_list =[]

seed(1)

for cat_variable in features:
        coef, p = spearmanr(df['satisfaction'], df[cat_variable])
        # calculate kendall's correlation
        # interpret the significance
        alpha = 0.05
        if p > alpha:
            message ='Samples are uncorrelated (fail to reject H0)'
        else:
            message='Samples are correlated (reject H0)'
        
        cat_variable_list.append(cat_variable)
        coef_list.append(round(coef,3))
        p_list.append(round(p,3))
        message_list.append(message)

out= pd.DataFrame({'variable': cat_variable_list, 'p':p_list, 'coef': coef_list, 'comment':message_list })      

spearmans_df= df[['satisfaction','compensation_satisfaction',
       'motivation', 'growth', 'fairness', 'flexibility', 'full_capacity',
       'overtime_working_freq', 'learning', 'discrepancy_promised_received',
       'sense_of_belonging', 'values_alignement', 'working_env',
       'supervisors_support', 'joy', 'security', 'searching_new_job']].copy()

plt.figure(figsize = (16,10))

corr = spearmans_df.corr(method = 'spearman')
sns.heatmap(corr)
fig = plt.figure(figsize = (15,100))
plt.show()

## Kendall's tau

cat_variable_list =[]
coef_list =[]
p_list =[]
message_list =[]

for cat_variable in features:
    coef, p = kendalltau(df['satisfaction'], df[cat_variable])
    # interpret the significance
    alpha = 0.05
    if p > alpha:
        message ='Samples are uncorrelated (fail to reject H0)'
    else:
        message='Samples are correlated (reject H0)'

    cat_variable_list.append(cat_variable)
    coef_list.append(round(coef,3))
    p_list.append(round(p,3))
    message_list.append(message)

out= pd.DataFrame({'variable': cat_variable_list, 'p':p_list, 'coef': coef_list, 'comment':message_list })      

plt.figure(figsize = (16,10))

corr = spearmans_df.corr(method='kendall')
sns.heatmap(corr)

plt.show()

## Kruskal-Wallis Test

df = df[['satisfaction','sex','age', 'education', 'family_status', 'current_role', 'experience_yrs', 'company_size', 'company_level',  'company_type']]

features =['age', 'sex', 'education', 'family_status', 'current_role', 'experience_yrs', 'company_size', 'company_level',  'company_type']

kruskal_results = []
message_list =[]
cat_variable_list= []
stats_list =[]
p_list =[]
# Iterate through the columns and perform the Kruskal-Wallis test
for column in features:
    filtered_data = df.dropna(subset=[column, 'satisfaction']) 
    
    # Group the 'satisfaction' values by unique values in 'column'
    groups = filtered_data[column].unique()
    group_data = [filtered_data[filtered_data[column] == group]['satisfaction'] for group in groups]
    
    # Perform the Kruskal-Wallis test only if there are more than one group
    if len(group_data) > 1:
        stat, p = kruskal(*group_data)
            
        cat_variable_list.append(column)
        stats_list.append(round(stat,3))
        p_list.append(round(p,3))
    else:
        cat_variable_list.append(column)
        stats_list.append(round(stat,3))
        p_list.append(np.nan)
    
    alpha = 0.05
    if p < alpha:
        message_list.append("There is a significant difference between the groups (reject H0)")
    else:
        message_list.append("There is no significant difference between the groups (fail to reject H0)")

out= pd.DataFrame({'variable': cat_variable_list, 'p':p_list, 'stats': stats_list, 'comment': message_list})      

## Cronbach alpha

df = pd.read_excel('it_dataset_recoded.xlsx')


# Calling function to the calculate value of Cronbach's alpha
cronbach_alpha(df[['satisfaction', 'searching_new_job', 'compensation_satisfaction','motivation', 'growth', 'fairness', 'flexibility', 'full_capacity', 'overtime_working_freq', 'learning' ,'discrepancy_promised_received', 'sense_of_belonging', 'values_alignement', 'working_env', 'supervisors_support', 'joy', 'security']])
