from scipy import stats
import numpy as np

p = [0.45, 0.3, 0.4, 0.1, 0.25]

#pull arm
def pull(arm):
    if np.random.rand() < p[arm]:
        return 1
    return 0

wins = [0,0,0,0,0]
pulls = [0,0,0,0,0]
n = 10000

for run in range(0, n):
    priors = [stats.beta(a=1+win, b=1+pull-win) for pull, win in zip(pulls, wins)] 
    theta = [sample.rvs(1) for sample in priors]
    choice = np.argmax(theta)
    current_pull = pull(choice)
    pulls[choice] += 1
    wins[choice] += current_pull
    if pulls[0] & pulls[1] & pulls[2] & pulls[3] & pulls[4]:
        print(run, wins[0]/pulls[0], wins[1]/pulls[1], wins[2]/pulls[2], wins[3]/pulls[3], wins[4]/pulls[4])
