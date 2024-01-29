import numpy as np
from bayes_opt import BayesianOptimization


# and plasma power
def deposition_model(temp, power):
    rate = 5 + 0.1*temp + 0.2*power - 0.01*(temp-300)**2
    return rate

# Bayesian optimization to maximize deposition rate by 
# tuning temperature (100 to 500 C) and power (50 to 150 W)
bo = BayesianOptimization(deposition_model, {'temp':(100, 500),  
                                             'power':(50, 150)})  

bo.maximize(init_points=10, n_iter=15)

print(bo.max) # Print optimized deposition rate
print(bo.max['params']) # Print optimal parameters