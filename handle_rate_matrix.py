if __name__ == "__main__":
    import numpy as np
    save_file = "rate_matrixs.dat"
    #rate_matrixs = np.fromfile(save_file, dtype=np.float, sep=' ').reshape(n_img, n_model, -1)
    n_model = 6
    rate_matrixs = np.fromfile(save_file, dtype=np.float, sep=' ')
    n_img = rate_matrixs.size // n_model // 4
    rate_matrixs = rate_matrixs.reshape(n_img, n_model, -1)
    for i in range(n_model):
        mat = np.vstack([rate[i] for rate in rate_matrixs])
        np.savetxt('rate_matrix_{}.txt'.format(i), mat, fmt='%.4f')