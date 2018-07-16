
import numpy as np
import pandas as pd
import scipy.stats as stats

#Create dataframe first
Hs=np.array([60,40,100])
Bs =np.array([54,44,98])
Ms=np.array([46,53,99])
Pd=np.array([41,57,98])
Totl=np.array([201,194,395])

Dataset= pd.DataFrame({"High School":Hs,"Bachelors":Bs, "Masters":Ms, "Ph.d.":Pd, "Total":Totl})
Dataset.index =["Female","Male","Total"]
#Dataset is created for data analysis

observed = Dataset.ix[0:2,0:4]   # Get table without totals for later use
print("*"*15,"Dataset","*"*15,"\n",Dataset)
#Create Expected Dataset
expected =  np.outer(Dataset["Total"][0:2],
                     Dataset.ix["Total"][0:4]) / 395
expected = pd.DataFrame(expected)
                     
expected.columns = ["High School","Bachelors", "Masters", "Ph.d."]
expected.index = ["Female","Male"]   
#Expected Dataset Created
print("*"*15,"Expected Dataset","*"*15,"\n",expected)

chi_squared_stat = (((observed-expected)**2)/expected).sum().sum()

print("\n Chi_Squared _stat: ",chi_squared_stat)      
print("\n *Note: We call .sum() twice: once to get the column sums and a second time to add the column sums together, returning the sum of the entire 2D table.")

crit = stats.chi2.ppf(q = 0.95, # Find the critical value for 5% of segnificant level*
                      df = 8)   # *

print("\n\nCritical value")
print(crit)

p_value = 1 - stats.chi2.cdf(x=chi_squared_stat,  # Find the p-value
                             df=8)
print("\nP value")
print(p_value)
print("\nDf=3")          
print("""\n*Note: The degrees of freedom for a test of independence equals the product of the number of categories in each variable minus 1. In this case we have a 4x2 table so df = 3x1 = 3.
As with the goodness-of-fit test, we can use scipy to conduct a test of independence quickly. Use stats.chi2_contingency() function to conduct a test of independence automatically given a frequency table of observed counts:
      """)   

print("\n",stats.chi2_contingency(observed= observed))

print("From above we conclude  that the education level depends on gender at a 5% level of significance.")