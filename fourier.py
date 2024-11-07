import numpy as np
import cupy as cp
import scipy
import cupyx
from tqdm import tqdm


#fouier処理とダウンサンプリングのコード
def LPF_3D(data, kc, ap, gaussianfilter=True, limit_memory=1536*1536*128):

    raw_resolution = data.shape
    x_down_GS = np.zeros((raw_resolution[0],kc*2,kc*2), dtype=np.complex64)
    x_GS = np.zeros(raw_resolution, dtype=np.complex64)
    roop = int(data.size/limit_memory)

    sigma = [data.shape[0]/ap/2, data.shape[1]/kc/2, data.shape[2]/kc/2]
    if roop==0:
        roop = 1
    for n in range(roop):
        if n<(roop-1):
            data[int(raw_resolution[0]/roop)*n:int(raw_resolution[0]/roop)*(n+1)] = cp.asnumpy(cupyx.scipy.ndimage.gaussian_filter1d(cp.asarray(data[int(raw_resolution[0]/roop)*n:int(raw_resolution[0]/roop)*(n+1)]), sigma=sigma[0], axis=0, mode="reflect"))
        else:
            data[int(raw_resolution[0]/roop)*n:] = cp.asnumpy(cupyx.scipy.ndimage.gaussian_filter1d(cp.asarray(data[int(raw_resolution[0]/roop)*n:]), sigma=sigma[0], axis=0, mode="reflect"))

    # data = scipy.ndimage.gaussian_filter1d(data, sigma=sigma[0], axis=0, mode="reflect")
    sigma[0] = 0


    for n in range(roop):

        if n<(roop-1):

            if gaussianfilter:
                x = cupyx.scipy.ndimage.gaussian_filter(cp.asarray(data[int(raw_resolution[0]/roop)*n:int(raw_resolution[0]/roop)*(n+1)]), sigma=sigma, mode="wrap")
                # x = cp.asarray(scipy.ndimage.gaussian_filter(data[int(raw_resolution[0]/roop)*n:int(raw_resolution[0]/roop)*(n+1)], sigma=sigma, mode="wrap"))
            else:
                x = cp.asarray(data[int(raw_resolution[0]/roop)*n:int(raw_resolution[0]/roop)*(n+1)])

            # X_gpu = cp.fft.fftn(cp.asarray(data[int(raw_resolution[0]/roop)*n:int(raw_resolution[0]/roop)*(n+1)]), axes=(1,2))
            X_gpu = cp.fft.fftn(x, axes=(1,2))
            X_gpu[:,kc+1:X_gpu.shape[1]-kc] = 0
            X_gpu[:,:,kc+1:X_gpu.shape[2]-kc] = 0

            x_GS[int(raw_resolution[0]/roop)*n:int(raw_resolution[0]/roop)*(n+1)] = cp.asnumpy(cp.fft.ifftn(X_gpu, axes=(1,2)))
            
            X_down_GS_gpu = cp.delete(X_gpu, slice(kc+1,raw_resolution[1]-kc+1), axis=1)
            del X_gpu
            cp.get_default_memory_pool().free_all_blocks()
            X_down_GS_gpu = cp.delete(X_down_GS_gpu, slice(kc+1,raw_resolution[2]-kc+1), axis=2)
            X_down_GS_gpu[:,kc] = X_down_GS_gpu[:,kc] + X_down_GS_gpu[:,kc].conjugate()
            X_down_GS_gpu[:,:,kc] = X_down_GS_gpu[:,:,kc] + X_down_GS_gpu[:,:,kc].conjugate()
            X_down_GS_gpu = cp.fft.ifftn(X_down_GS_gpu, axes=(1,2)) / raw_resolution[1] * kc*2 / raw_resolution[2] * kc*2
            x_down_GS[int(raw_resolution[0]/roop)*n:int(raw_resolution[0]/roop)*(n+1)] = cp.asnumpy(X_down_GS_gpu)

        else:

            if gaussianfilter:
                x = cupyx.scipy.ndimage.gaussian_filter(cp.asarray(data[int(raw_resolution[0]/roop)*n:]), sigma=sigma, mode="wrap")
                # x = cp.asarray(scipy.ndimage.gaussian_filter(data[int(raw_resolution[0]/roop)*n:], sigma=sigma, mode="wrap"))
            else:
                x = cp.asarray(data[int(raw_resolution[0]/roop)*n:])

            X_gpu = cp.fft.fftn(x, axes=(1,2))
            X_gpu[:,kc+1:X_gpu.shape[1]-kc] = 0
            X_gpu[:,:,kc+1:X_gpu.shape[2]-kc] = 0

            x_GS[int(raw_resolution[0]/roop)*n:] = cp.asnumpy(cp.fft.ifftn(X_gpu, axes=(1,2)))
            # x_GS = gaussian_filter1d(x_GS, int(x_GS.shape[0]/ap/4), axis=0, truncate=4)

            X_down_GS_gpu = cp.delete(X_gpu, slice(kc+1,raw_resolution[1]-kc+1), axis=1)
            del X_gpu
            cp.get_default_memory_pool().free_all_blocks()
            X_down_GS_gpu = cp.delete(X_down_GS_gpu, slice(kc+1,raw_resolution[2]-kc+1), axis=2)
            X_down_GS_gpu[:,kc] = X_down_GS_gpu[:,kc] + X_down_GS_gpu[:,kc].conjugate()
            X_down_GS_gpu[:,:,kc] = X_down_GS_gpu[:,:,kc] + X_down_GS_gpu[:,:,kc].conjugate()
            X_down_GS_gpu = cp.fft.ifftn(X_down_GS_gpu, axes=(1,2)) / raw_resolution[1] * kc*2 / raw_resolution[2] * kc*2
            x_down_GS[int(raw_resolution[0]/roop)*n:] = cp.asnumpy(X_down_GS_gpu)

    x_down_GS = x_down_GS.reshape(ap*2, int(x_down_GS.shape[0]/ap/2), x_down_GS.shape[1], x_down_GS.shape[2]).mean(axis=(1))
    
    return x_down_GS, x_GS


def SGS_stress(data, data_GS):

    data_SGS = {name:np.zeros(data[name].shape, dtype=np.complex64) for name in ["vx", "vy", "vz"]}
    for name in ["vx", "vy", "vz"]:
        data_SGS[name] = data[name] - data_GS[name]

    SGS_stress_tensor = {"vxvx":0}
    for n, name1 in enumerate(["vx", "vy", "vz"]):
        for name2 in tqdm(["vx", "vy", "vz"][n:]):
            SGS_stress_tensor[name1+name2] = data_SGS[name1] * np.conjugate(data_SGS[name2])

    return SGS_stress_tensor


# def favre_LPF_3D():



