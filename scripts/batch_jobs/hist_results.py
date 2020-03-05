import numpy as np
import matplotlib.pyplot as plt
import sys

accuracy_list = np.loadtxt(sys.argv[1],dtype='float', delimiter=',')
#header is accuracy, TT, FF, sum

#accuracy_list *= 100
#accuracy_list = ['{:2.2f}'.format(x) for x in accuracy_list]
#accuracy_list = np.array(accuracy_list).astype(dtype='float')
print(accuracy_list)
plt.hist(accuracy_list[:,0], bins=50)
plt.show()
