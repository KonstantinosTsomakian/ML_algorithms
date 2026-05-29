
#Part a
# Define probabilities
p_photon = 1e-7
p_photon

false_p = 0.1

true_p = 0.85

prior = p_photon

marginal = true_p * p_photon + false_p * (1 -p_photon)

# Compute probability using the Bayes theorem
p_photon_given_detection = (true_p * prior) / marginal

print('The answer for question 2a is : {}'.format(p_photon_given_detection))

#Part b
import numpy as np
import matplotlib.pyplot as plt


energies = [10,20,30,40]
prob_energy = [0.25, 0.25, 0.25, 0.25]

n = 100
its = 2000


#Compute package energies
total = [np.sum(np.random.choice(energies, n, p=prob_energy)) for i in range(its)]


#plot the results
plt.figure(figsize=(10,8))
plt.hist(total,
          bins = 30,
          color='pink',
          edgecolor = 'black',
          density = True)
plt.xlabel('Total Energy'),
plt.ylabel('PDF'),
plt.grid(True)
plt.show()





#Assuming a gaussian i can compute with MLE the estimated mean and the estimated standard deviation and compare the mean with the sample mean and the sample std.
sample_mean = np.mean(total)
sample_std = np.std(total, ddof=1)


gaussian_mean = np.mean(total)
gaussian_std = np.std(total, ddof= 0)
print('The  sample mean and the sample standard deviation are {} and {}'.format(sample_mean, sample_std))
print('The  gaussian mean and the gaussian standard deviation are {} and {}'.format(gaussian_mean, gaussian_std))




# Part c

draws = 10000
mean = 1e-7
std = 9e-8

#Draw prior probabilities and keep only the positive ones
p_photon_new = np.random.normal(mean, std, size = draws)
p_photon_new = p_photon_new[p_photon_new > 0]

# Apply the Bayes theorem
prior_new = p_photon_new

marginal_new = true_p * p_photon + false_p * (1 -p_photon)

p_photon_given_detection_new = (true_p * prior_new) / marginal_new



#Plot the resulting values
plt.figure(figsize=(10,8))
plt.hist(p_photon_given_detection_new,
          bins = 30,
          color='pink',
          edgecolor = 'black',
          density = True)
plt.xlabel('P(report | package)'),
plt.ylabel('PDF'),
plt.grid(True)
plt.show()