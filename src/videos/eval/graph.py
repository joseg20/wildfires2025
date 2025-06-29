import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

choosen_wildfires = [
    "20180504_FIRE_smer-tcs8-mobo-c",
    "20180726_FIRE_so-w-mobo-c",
    "20200930_inMexico_lp-s-mobo-c",
    "20180809_FIRE_mg-w-mobo-c",
    "20200822_BrattonFire_lp-e-mobo-c",
    "20200905_ValleyFire_lp-n-mobo-c",
    "20160722_FIRE_mw-e-mobo-c",
    "20190716_FIRE_bl-s-mobo-c",
    "20180603_FIRE_sm-w-mobo-c",
    "20180806_FIRE_mg-s-mobo-c",
    "20170625_BBM_bm-n-mobo",
    "20200806_SpringsFire_lp-w-mobo-c",
    "20201127_Hawkfire_pi-w-mobo-c",
    "20200911_FIRE_mlo-s-mobo-c",
    "20200705_FIRE_bm-w-mobo-c",
    "20200829_inside-Mexico_mlo-s-mobo-c",
    "20170708_Whittier_syp-n-mobo-c",
    "20200611_skyline_lp-n-mobo-c",
    "20190610_FIRE_bh-w-mobo-c",
    "20200601_WILDLAND-DRILLS_om-e-mobo-c",
    "20160711_FIRE_ml-n-mobo-c",
    "20190924_FIRE_sm-n-mobo-c",
    "20170520_FIRE_lp-s-iqeye",
]

experiment_name = "experiments/example/all_states.json"

with open(f"experiments/{experiment_name}/all_states.json", "r") as f:
    incendios2 = json.load(f)

n = 20
fires = len(choosen_wildfires)
global_true_labels = [] 
global_predicted_labels = []  

filtered_incendios2 = sorted(
    [incendio for incendio in incendios2 if incendio[0] in choosen_wildfires],
    key=lambda x: choosen_wildfires.index(x[0])
)

filtered_incendios2 = filtered_incendios2[:fires]

cmap = ListedColormap(['#fefebd', '#a50026', '#045837'])

fig, ax = plt.subplots(fires, 1, figsize=(26, 28), sharex=True)
i = 0
for incendio in filtered_incendios2:
    nombre = incendio[0]
    detecciones = np.array(incendio[1:])
    
    idx_cero = np.where(detecciones == 0)[0]
    if len(idx_cero) == 0:
        continue
    idx_cero = idx_cero[0]  
    
    izq = max(0, idx_cero - n)
    der = min(len(detecciones), idx_cero + n + 1)
    
    ventana = detecciones[izq:der]
    
    if len(ventana) < 2 * n + 1:
        padding_izq = max(0, n - (idx_cero - izq))
        padding_der = max(0, n - (der - idx_cero - 1))
        ventana = np.pad(ventana, (padding_izq, padding_der), mode='constant', constant_values=0)
    
    ventana_colores = np.array([0 if x == 0 else (2 if x == 1 else 1) for x in ventana])
    
    ax[i].imshow([ventana_colores], cmap=cmap, aspect='auto')

    ax[i].set_yticks([0])
    ax[i].set_yticklabels([nombre], fontsize=28, fontweight='bold') 
    
    num_ticks = len(ventana)
    x_tick_labels = np.arange(-n, n + 1)
    ax[i].set_xticks(np.arange(num_ticks))
    ax[i].set_xticklabels(x_tick_labels, fontsize=16, fontweight='bold')
    ax[i].set_xlim(0, num_ticks - 1)
    
    true_labels_before = [0] * (idx_cero - izq)
    predicted_labels_before = list(ventana[:idx_cero - izq])
    true_labels_after = [1] * (der - idx_cero)
    predicted_labels_after = [1 if x == 1 else 0 for x in ventana[idx_cero - izq:]]
    
    predicted_labels_before = [0 if x == 1 else 1 for x in ventana[:idx_cero - izq]]
    
    true_labels = true_labels_before + true_labels_after
    predicted_labels = predicted_labels_before + predicted_labels_after
    min_len = min(len(true_labels), len(predicted_labels))
    true_labels = true_labels[:min_len]
    predicted_labels = predicted_labels[:min_len]
    global_true_labels.extend(true_labels)
    global_predicted_labels.extend(predicted_labels)
    i += 1
    if i == fires:
        break

plt.xlabel("Frames (relative to frame 0)", fontsize=28, fontweight='bold')
plt.subplots_adjust(left=0.37, top=0.999, bottom=0.03, right=0.94, hspace=0.4)
plt.savefig(f"predictions_{experiment_name}.png")
