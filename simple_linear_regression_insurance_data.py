import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Load the data:


df_insurance = pd.read_csv('insurance.csv')

# Linear regression with bmi & charges

# plottting bmi vs. charges reveals a not-so-linear relationship


df_insurance.plot(kind='scatter', x = 'bmi', y = 'charges', c='black', s=20)

# the next two lines maximize the figure's window
mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())

plt.show()



# but their appear to be two clusters that might be linear
# let's check only males


ax=df_insurance[df_insurance['sex'] == 'male'].plot(kind='scatter', x = 'bmi', y = 'charges', color = 'purple', label = 'male')
df_insurance[df_insurance['sex'] == 'female'].plot(kind='scatter', x = 'bmi', y = 'charges', color = 'orange', ax=ax, label = 'female')
plt.legend(loc='upper left')

mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.show()

# separating by male and female didn't separate the clusters. let's
# see if the clusters are related to smoking


ax=df_insurance[(df_insurance['smoker'] == 'yes')].plot(kind='scatter', x = 'bmi', y = 'charges', color = 'b', label = 'smoker')

df_insurance[(df_insurance['smoker'] == 'no')].plot(kind='scatter', x = 'bmi', y = 'charges', color = 'g', label = 'non-smoker', ax=ax)

plt.legend(loc='upper left')

mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.show()

# success!  each cluster looks somewhat linear, so let us regress

from sklearn.linear_model import LinearRegression
model = LinearRegression()

bmi_smoker = df_insurance[df_insurance['smoker'] == 'yes']['bmi'].values.reshape(-1,1)
charges_smoker = df_insurance[df_insurance['smoker'] == 'yes']['charges'].values.reshape(-1,1)

bmi_nonsmoker = df_insurance[df_insurance['smoker'] == 'no']['bmi'].values.reshape(-1,1)
charges_nonsmoker = df_insurance[df_insurance['smoker'] == 'no']['charges'].values.reshape(-1,1)



fig = plt.figure()
fig.suptitle('bmi vs. charges, smoker & non-smoker, linear regression', fontsize = 20)
ax = fig.add_subplot(111)
ax.set_xlabel('bmi')
ax.set_ylabel('charges')

reg = LinearRegression().fit(bmi_smoker, charges_smoker)
ax.scatter(bmi_smoker,charges_smoker, color='b', label='smoker', s=25)
ax.plot(bmi_smoker, reg.predict(bmi_smoker), color='b', linewidth=3,
        label = 'y = ' + str(round(reg.coef_[0][0],2)) + 'x + ' +
        str(round(reg.intercept_[0],2)))


# plt.show()

reg = LinearRegression().fit(bmi_nonsmoker, charges_nonsmoker)

ax.scatter(bmi_nonsmoker,charges_nonsmoker, color='g', label='non-smoker', s=25)
ax.plot(bmi_nonsmoker, reg.predict(bmi_nonsmoker), color='g',
        linewidth=3, label = 'y = ' + str(round(reg.coef_[0][0],2)) +
        'x + ' + str(round(reg.intercept_[0],2)))

plt.legend(loc='upper left')

mng = plt.get_current_fig_manager()
mng.resize(*mng.window.maxsize())
plt.show()


# things to do:

# 1. get p-value
# 2. split data into train and test
#  a. regress training
#  b. test
#  c. analyize
