import matplotlib.pyplot as plt
import numpy as np

all_n = [1, 2, 3, 4, 5, 10, 15, 20, 25]

alex_fmax = [52.45793901, 62.7891693, 68.15194532, 72.06887487, 75.19716088, 83.04416404, 86.85594111, 89.26130389,
             90.86487907]
alex_fvlad = [67.86, 76.77, 81.11, 84.41, 86.51, 91.23, 93.52, 94.98, 95.78]
alex_nopool = [74.96056782, 82.41324921, 85.72555205, 87.64458465, 89.09043113, 92.77076761,
               94.41377497, 95.36014721, 96.28023134]

vgg16_fmax = [79.63985279, 85.90956887, 88.91955836, 90.56256572, 91.65352261, 94.61093586,
              95.66246057, 96.42481598, 97.16088328]
vgg16_fvlad = [75.92008412, 84.06940063, 87.63144059, 89.77392219, 91.08832808, 94.99211356,
               96.49053628, 97.33175605, 97.81808623]
vgg16_nopool = [81.79547844, 88.26235542, 90.68086225, 92.24500526, 93.28338591, 95.72818086,
                96.6745531, 97.25289169, 97.77865405]

alex_fmax_shelf = [32.9915878, 42.73133544, 48.44900105, 52.77339642, 56.42744479, 67.41587802,
                   73.31756046, 77.33964248, 80.58622503]
alex_fvlad_shelf = [53.26, 62.39, 67.35, 70.85, 73.26, 80.80, 84.86, 87.20, 88.89]
alex_nopool_shelf = [53.11514196, 61.34332282, 65.66771819, 68.87486856, 71.25394322, 78.43059937,
                     81.99263933, 84.16140904, 85.94900105]

vgg16_fmax_shelf = [43.95373291, 53.47003155, 59.22712934, 63.26235542, 66.18033649, 75.223449,
                    80.29705573, 83.80651945, 86.35646688]
vgg16_fvlad_shelf = [69.95268139, 78.61461619, 82.58412198, 85.29179811, 87.40799159, 91.75867508,
                     93.79600421, 95.14984227, 95.84647739]
vgg16_nopool_shelf = [50.88065195, 58.67507886, 62.85488959, 65.5362776, 67.53417455, 74.46109359,
                      78.94321767, 81.88748686, 83.75394322]

plt.style.use('seaborn-bright')

plt.figure()
s = 8
plt.plot(all_n, vgg16_nopool, markerfacecolor='none', markersize=s, color='#ff00ff', marker='^', label=f"no pool (VGG16)")
plt.plot(all_n, vgg16_fmax, markerfacecolor='none', markersize=s, color='#ff00ff', marker='x', label=f"fmax (VGG16)")
plt.plot(all_n, vgg16_fvlad, markerfacecolor='none', markersize=s, color='#ff00ff', marker='o', label=f"fvlad (VGG16)")
plt.plot(all_n, alex_nopool, markerfacecolor='none', markersize=s, color='#ff0000', marker='^', label=f"no pool (Alex)")
plt.plot(all_n, alex_fvlad, markerfacecolor='none', markersize=s, color='#ff0000', marker='o', label=f"fvlad (Alex)")
plt.plot(all_n, alex_fmax, markerfacecolor='none', markersize=s, color='#ff0000', marker='x', label=f"fmax (Alex)")

plt.plot(all_n, vgg16_nopool_shelf, markerfacecolor='none', markersize=s, color='#00f   f00', marker='^',
         label=f"no pool (VGG16) shelf")
plt.plot(all_n, vgg16_fmax_shelf, markerfacecolor='none', markersize=s, color='#00ff00', marker='x',
         label=f"fmax (VGG16) shelf")
plt.plot(all_n, vgg16_fvlad_shelf, markerfacecolor='none', markersize=s, color='#00ff00', marker='o',
         label=f"fvlad (VGG16) shelf")
plt.plot(all_n, alex_nopool_shelf, markerfacecolor='none', markersize=s, color='#0000ff', marker='^',
         label=f"no pool (Alex) shelf")
plt.plot(all_n, alex_fmax_shelf, markerfacecolor='none', markersize=s, color='#0000ff', marker='x',
         label=f"fmax (Alex) shelf")
plt.plot(all_n, alex_fvlad_shelf, markerfacecolor='none', markersize=s, color='#0000ff', marker='o',
         label=f"fvlad (Alex) shelf")

plt.ylabel('recall@N (%)')
plt.xlabel('top N results from database')
plt.ylim(0, 100)
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
