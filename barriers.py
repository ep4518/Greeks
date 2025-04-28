import RR as rr
import BS as bs
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from BSPricer.BSModel01 import BSModel, MJDModel
from BSPricer.PathDepOption01 import ArthmAsianCall, EuropeanCall, CallUpOut, PutDownIn

S = np.linspace(20, 300, 281) / 100
K = 1.0
H = 0.7 
R = 0.05
T = 1.0
r = 0.0
q = 0.0
sigma = 0.2


cuo = rr.CUO(S, K, H, R, T, r, q, sigma)
c = bs.C(S, K, T, r, q, sigma)

# px.line(y=cuo, x=S).show()
# px.line(y=c, x=S).show()

pdi = -rr.PDI(S, K, H=H, R=R, T=T, r=r, q=q, sigma=sigma)
pdi = pd.Series(pdi, index=S)
px.line(pdi).show()

model = BSModel(1.0, r, sigma)
monte_cuo = CallUpOut(T, K, H, R, int(252 * T))

res = monte_cuo.PriceByMC(model, 1000, 0.001)

V = {}
for S in np.arange(0.2, 3.05, 0.05):
    V[S] = PutDownIn(T, K, H, R, int(252 * T), European=True).PriceByMC(BSModel(S, r, sigma), 1000, 0.0001)

V = pd.Series(V).sort_index()
MC = -V
fig, ax = plt.subplots()

# Plot the lines with labels and colors
ax.plot(pdi, label='Reiner-Rubinstein', color='purple', linewidth=2, linestyle='-')
ax.plot(MC, label='Monte-Carlo', color='black', linewidth=2, linestyle='dotted')

# Add title and axis labels
ax.set_title(f'Comparison of Reiner-Rubinstein and Monte Carlo for PDI\nT: {T}, K: {K}, B: {H}, R: {R}, r: {r}, q: {q}, sigma: {sigma}, MC model: {model.__class__.__name__}', fontsize=14)
ax.set_xlabel('S', fontsize=12)
ax.set_ylabel('Value', fontsize=12)

# Add grid and legend
ax.grid(True, linestyle=':', alpha=0.7)
ax.legend()

# Optional: tighter layout
# fig.tight_layout()

# Display the figure
plt.show()