import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.plot([1,2,3])
plt.savefig('test2.png',bbox_inches = 'tight',
    pad_inches = 0)