import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import random

total_views = 10000
num_ads = 5
ad_list = []
total_reward = 0
for view in range(total_views):
    ad = random.randrange(num_ads)
    ad_list.append(ad)
    reward = dataset.values[view, ad]
    total_reward = total_reward + reward
