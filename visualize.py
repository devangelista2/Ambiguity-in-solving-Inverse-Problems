import matplotlib.pyplot as plt
import numpy as np

noise_level = 0.0
delta = 0.03 # Out-of-domain noise intensity

if noise_level == 0:
    suffix = '0'
elif noise_level == 0.025:
    suffix = '025'

errors = np.load(f'relerrors_ssnet_{suffix}_{delta}.npy')
xaxis = np.linspace(0, delta, 11) #* 256

plt.figure()
plt.plot(xaxis, errors[0],'o-')
plt.plot(xaxis, errors[1],'o-')
plt.plot(xaxis, errors[2],'o-')
plt.grid()
plt.legend(['NN', 'FiNN', 'StNN'])
plt.xlabel(r'$\delta$')
plt.ylabel(r'$\|\| \Psi(Kx + e) - x \|\|$')
plt.axis(ymin=0.05, ymax=0.30)
# plt.tight_layout()
# plt.title('Test case A.1')
plt.xticks(xaxis, rotation=30)
plt.savefig(f'relerrors_ssnet_{suffix}_{delta}_EE.png', dpi=300)