import numpy as np
import BS as bs
import RR as rr
import matplotlib.pyplot as plt

B: float = 0.7
R: float = 0.05
K: float = 1.0
r: float = 0.0
q: float = 0.0
sigma: float = 0.3

S = np.linspace(20, 300, 281) / 100
T = np.linspace(0.01, 5, 9801)

X, Y = np.meshgrid(S, T)

pnl = bs.P(X, K, Y, r, q, sigma) + \
    bs.P(X, K * 0.73, Y, r, q, sigma) - \
    bs.P(X, K * 0.7, Y, r, q, sigma)
delta = bs.deltaP(X, K, Y, r, q, sigma) + \
    bs.deltaP(X, K * 0.73, Y, r, q, sigma) - \
    bs.deltaP(X, K * 0.7, Y, r, q, sigma)
gamma = bs.gamma(X, K, Y, r, q, sigma) + \
    bs.gamma(X, K * 0.73, Y, r, q, sigma) - \
    bs.gamma(X, K * 0.7, Y, r, q, sigma)
vega = bs.vega(X, K, Y, r, q, sigma) + \
    bs.vega(X, K * 0.73, Y, r, q, sigma) - \
    bs.vega(X, K * 0.7, Y, r, q, sigma)
theta = bs.thetaP(X, K, Y, r, q, sigma) + \
    bs.thetaP(X, K * 0.73, Y, r, q, sigma) - \
    bs.thetaP(X, K * 0.7, Y, r, q, sigma)
rho = bs.rhoP(X, K, Y, r, q, sigma) + \
    bs.rhoP(X, K * 0.73, Y, r, q, sigma) - \
    bs.rhoP(X, K * 0.7, Y, r, q, sigma)

fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(231, projection='3d')
ax2 = fig.add_subplot(232, projection='3d')
ax3 = fig.add_subplot(233, projection='3d')
ax4 = fig.add_subplot(234, projection='3d')
ax5 = fig.add_subplot(235, projection='3d')
ax6 = fig.add_subplot(236, projection='3d')

surf = ax1.plot_surface(X, Y, pnl, cmap='viridis', edgecolor='none')
surf = ax2.plot_surface(X, Y, delta, cmap='viridis', edgecolor='none')
surf = ax3.plot_surface(X, Y, gamma, cmap='viridis', edgecolor='none')
surf = ax4.plot_surface(X, Y, vega, cmap='viridis', edgecolor='none')
surf = ax5.plot_surface(X, Y, theta, cmap='viridis', edgecolor='none')
surf = ax6.plot_surface(X, Y, rho, cmap='viridis', edgecolor='none')

for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.set_xlabel('Spot Price (S)')
    ax.set_ylabel('Time to Maturity (T)')
    ax.set_zlabel('Value')

ax1.set_title('Black-Scholes Put Price Surface')
ax2.set_title('Black-Scholes Put Delta Surface')
ax3.set_title('Black-Scholes Put Gamma Surface')
ax4.set_title('Black-Scholes Put Vega Surface')
ax5.set_title('Black-Scholes Put Theta Surface')
ax6.set_title('Black-Scholes Put Rho Surface')

# fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

plt.tight_layout()
plt.show()


fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(projection='3d')
pdo = rr.CUO(X, K, 1.3, R, Y, r, q, sigma)
surf = ax.plot_surface(X, Y, pdo, cmap='viridis', edgecolor='none')
plt.show()
