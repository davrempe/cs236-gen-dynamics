import numpy as np
import matplotlib.pyplot as plt
import os

OUT_DIR = './final_plots'

pos_err = [0.,         0.00342099, 0.00662011, 0.00995325, 0.014036,   0.01884641,
 0.02405123, 0.02963962, 0.03559843, 0.04189672, 0.04848562, 0.0553606,
 0.06250975, 0.06985544, 0.07738324, 0.08509148, 0.09292993, 0.10086523,
 0.10889881, 0.11697846, 0.12502244, 0.13301681, 0.14091159, 0.14866642,
 0.15617752, 0.16333364, 0.17012209, 0.17645753, 0.18216289, 0.18719818,
 0.19158612, 0.19534244, 0.19837792, 0.20069723, 0.20248015, 0.20391539,
 0.20510976, 0.20617956, 0.20714497, 0.20803907, 0.20886644]
rot_err = [ 0.,          0.60225103,  1.24481251,  2.14412793,  3.25072329,  4.56742637,
  6.08226569,  7.75787439,  9.58773558, 11.54473864, 13.58605683, 15.72210737,
 17.95460956, 20.24547337, 22.56864941, 24.92185565, 27.30500628, 29.67505098,
 32.04055879, 34.39080907, 36.70809721, 38.95539968, 41.12258537, 43.19258084,
 45.13233242, 46.92177737, 48.54243338, 49.99174221, 51.25901614, 52.35099751,
 53.20220882, 53.82866871, 54.26548184, 54.55901239, 54.7570811,  54.89591214,
 54.99703712, 55.08167203, 55.14551955, 55.19391099, 55.23298483]

pos_err = np.array(pos_err) * 100.0 # want cm
rot_err = np.array(rot_err)

plot_suffix = '.png'

LINE_WIDTH=5.0
AXES_TITLE_SIZE=28
XTICK_SIZE=28
YTICK_SIZE=28
LEGEND_SIZE=26
TITLE_SIZE=36

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=TITLE_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=AXES_TITLE_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=XTICK_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=YTICK_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
plt.rc('figure', titlesize=TITLE_SIZE)  # fontsize of the figure title

plt.rcParams['font.family'] = 'Bahnschrift'

step_size = 1.0 / 6.0

f, axarr = plt.subplots(2, 1, figsize=(20, 10), dpi=300)
# pos
axarr[0].plot(np.arange(0, pos_err.shape[0])*step_size, pos_err, '-r', linewidth=LINE_WIDTH)
axarr[0].set(xlabel='Time Steps', ylabel='Pos Err (cm)')
axarr[0].set_title('Roll-out Error Over Time')
# rot
axarr[1].plot(np.arange(0, rot_err.shape[0])*step_size, rot_err, '-r', linewidth=LINE_WIDTH)
axarr[1].set(xlabel='Time (s)', ylabel='Rot Err (deg)')
for ax in axarr.flat:
    ax.label_outer()
    ax.grid(True)

plt.savefig(os.path.join(OUT_DIR, 'lstm_div' + plot_suffix))
