# Name: Bharghav Srikhakollu
# Date: 01-21-2023
#######################################################################################################
# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt

#######################################################################################################
# 1) A. Describe - To show minimum, maximum, mean and median values for each of 9 features
#######################################################################################################
# Read CSV file using Pandas: dataframe (df)
# Convert the missing values "99.0000" and "-99.0000" as "NaN", read only the 9 feature columns
df = pd.read_csv('cfhtlens.csv', na_values =(99.0000, -99.0000), usecols=[3, 4, 5, 6, 7, 8, 9, 10, 11])

output = df.describe().T
output = output.drop(['count', 'std', '25%', '75%'], axis = 1)
output.rename(columns = {'50%':'median'}, inplace = True)
print(output)
output.to_csv('Describe.csv')
#######################################################################################################
# 1) B. A histogram - For the 9 features
#######################################################################################################
df.hist(column=["PSF_e1", "PSF_e2", "scalelength", "model_flux", "MAG_u", "MAG_g", "MAG_r", "MAG_i", "MAG_z"],figsize = (10,10), color='green')
plt.savefig('Histogram.png')
#######################################################################################################
# 1) C. the number of missing values
#######################################################################################################
total = 0
for col in df.columns:
    total = df[col].isna().sum()
    print("The number of missing values for feature: " + col + " is " + str(total))
#######################################################################################################
