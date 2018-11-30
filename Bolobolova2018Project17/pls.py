import numpy as np
import os.path
import urllib.request
import zipfile
import glob

from io import BytesIO
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

def download_ecog(dir):
    file = '20100802S1_Epidural-ECoG+Food-Tracking_B_Kentaro+Shimoda_mat_ECoG64-Motion6'
    if os.path.exists(dir):
        return os.path.join(dir, file)

    url = 'http://neurotycho.brain.riken.jp/download/2012/%s.zip' % file
    with urllib.request.urlopen(url) as ECoG:
        ZippedECoG = zipfile.ZipFile(BytesIO(ECoG.read()))
        ZippedECoG.extractall(dir)

    return os.path.join(dir, file)

def get_ecog_and_motion(dir):
    n_ch = len(glob.glob(os.path.join(dir, 'ECoG_ch*.mat')))
    ECoG = []
    for ch in range(1, n_ch + 1):
        ECoGData = loadmat(os.path.join(dir, 'ECoG_ch%d.mat' % ch))
        ECoGData = ECoGData['ECoGData_ch%d' % ch]
        ECoG.append(ECoGData[0])

    ECoG = np.array(ECoG)

    Motion = loadmat(os.path.join(dir, 'Motion.mat'))
    Motion = Motion['MotionData']
    Motion = Motion[0][0]

    return ECoG, Motion

def get_data():
    dir = download_ecog('ECoG')
    ECoG, Motion = get_ecog_and_motion(dir)

    signals_for_motion = 1000 / 120    # ECoG recorded at 1KHz, while motion at 120Hz
    X, y = [], []
    for i in range(Motion.shape[0] - 1):
        start = int(i * signals_for_motion)
        X.append(ECoG[:, start:start + int(signals_for_motion)].flatten())
        y.append(Motion[i+1] - Motion[i])

    return train_test_split(X, y)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()
    pls = PLSRegression(n_components=2)
    pls.fit(X_train, y_train)
    print("test score is", pls.score(X_test, y_test))
