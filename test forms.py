import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import pandas as pd
from generate_forms.generate_spike import generate_spike
from generate_forms.generate_M import generate_M
from generate_forms.generate_CLDR_CRDL import generate_CLDR, generate_CRDL
from generate_forms.generate_parabola import generate_parabola

n_points = 100

base_params_m = {'start': 50, 'width': 50, 'height': 100}
variation_range_m = {'start': 100, 'width': 100, 'height': 100}

base_spike_params = {'start': 50, 'width': 50, 'height': 100}
variation_spike_range = {'start': 100, 'width': 100, 'height': 100}

base_cl_params = {'start': 50,'width': 50,'height': 100,'A': 0.5,'ratio': 0.5}
variation_cl_range = {'start': 100,'width': 100,'height': 100,'A': 1,'ratio': 1}

base_parabola_params = {'start': 50, 'width': 50, 'height': 100}
variation_parabola_range = {'start': 100, 'width': 100, 'height': 100}

m_form = generate_M(n_points, base_params_m)
spike_form = generate_spike(n_points, base_spike_params)
cldr_form = generate_CLDR(n_points, base_cl_params)
crdl_form = generate_CRDL(n_points, base_cl_params)
parabola_form = generate_parabola(n_points, base_parabola_params)

plt.figure(figsize=(10, 6))
plt.subplot(2, 3, 1)
plt.plot(m_form)
plt.title("M Form")

plt.subplot(2, 3, 2)
plt.plot(spike_form)
plt.title("Spike Form")

plt.subplot(2, 3, 3)
plt.plot(cldr_form)
plt.title("CLDR Form")

plt.subplot(2, 3, 4)
plt.plot(crdl_form)
plt.title("CRDL Form")

plt.subplot(2, 3, 5)
plt.plot(parabola_form)
plt.title("Parabola Form")

plt.tight_layout()
plt.show()
