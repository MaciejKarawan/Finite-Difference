#from FD_KO import *
from FD_KO_dividend import *
#from IF1.FD.heatmaps_plot_modified import *
import seaborn as sns

### Args
vol = 0.2
vol2 = [0.2, 0.3]
r = np.log(1 + 0.05)
option_type = 'e_put'
strike = 2200
expiration = 1
NAS = 400  # Number Asset Steps
s0 = 2194.05
barrier = 1900
top_barrier = 3


'''_, data_knock_uncertain, _, _ = FDB_with_memory(NAS, None, [0.2, 0.3], r, option_type, strike, expiration, barrier)
_, data_knock, _, _ = FDB_with_memory(NAS, None, [0.2, 0.2], r, option_type, strike, expiration, barrier)

#_, output_basic_wo_memory, init_basic_wo_memory = FD_wo_memory(NAS, s0, top_barrier, vol, r, option_type, strike, expiration)

sns.lineplot(data_knock_uncertain['return'], label='uncertain vol')
sns.lineplot(data_knock['return'], label='fixed vol')
plt.show()
'''

_, data_knock, _, grid_data = FDB_with_memory(NAS, s0, [0.2, 0.2], r, option_type, strike, expiration, barrier, True, 0.5, 200)
print(data_knock['return'], data_knock['theoretical'], data_knock['diff'])




'''sns.heatmap(grid_data['Grid'], cmap='magma')
plt.show()
'''

