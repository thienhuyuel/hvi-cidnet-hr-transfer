"""Mean intensity of the full-res teacher vs UHD-LL ground truth on the 30-pair subset -> results/uhd_brightness.txt"""
import glob, os, numpy as np, cv2
GT = '../Dataset/UHD-LL/testing_set/gt/'
a, b = [], []
for t in sorted(glob.glob('results/uhd_cache/teacher_*')):
    n = os.path.basename(t).split('teacher_')[1].rsplit('.', 1)[0] + '.JPG'
    a.append(cv2.imread(t).mean() / 255); b.append(cv2.imread(GT + n).mean() / 255)
s = f'n={len(a)} teacher mean {np.mean(a):.3f} gt mean {np.mean(b):.3f} teacher darker on {sum(x < y for x, y in zip(a, b))}/{len(a)}'
open('results/uhd_brightness.txt', 'w').write(s + '\n'); print(s)
