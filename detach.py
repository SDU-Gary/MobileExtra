import numpy as np
import OpenEXR
import Imath
import zarr
from zarr.storage import ZipStore



def write_one_exr(file_path, data):
    patch_data = {}
    """写入EXR文件"""
    header = OpenEXR.Header(data.shape[1], data.shape[0])
    header['channels'] = {name: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) for name in
                          {'Y'}}
    out_file = OpenEXR.OutputFile(file_path, header)

    for i, channel_name in enumerate(['Y']):
        patch_data[channel_name] = data[:, :, i].flatten().astype(np.float32).tobytes()
    out_file.writePixels(patch_data)
    out_file.close()

def write_RGB_exr(file_path, data):
    patch_data = {}
    """写入EXR文件"""
    header = OpenEXR.Header(data.shape[1], data.shape[0])
    header['channels'] = {name: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) for name in
                          {'R', 'G', 'B'}}
    out_file = OpenEXR.OutputFile(file_path, header)

    for i, channel_name in enumerate(['R', 'G', 'B']):
        patch_data[channel_name] = data[:, :, i].flatten().astype(np.float32).tobytes()
    out_file.writePixels(patch_data)
    out_file.close()

def write_two_exr(file_path, data):
    patch_data = {}
    """写入EXR文件"""
    header = OpenEXR.Header(data.shape[1], data.shape[0])
    header['channels'] = {name: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)) for name in
                          {'R', 'G'}}
    out_file = OpenEXR.OutputFile(file_path, header)

    for i, channel_name in enumerate(['R', 'G']):
        patch_data[channel_name] = data[:, :, i].flatten().astype(np.float32).tobytes()
    out_file.writePixels(patch_data)
    out_file.close()



def screen_space_position(w_position, pv, height, width):
    """Projects per-sample world-space positions to screen-space (pixel coordinates)

    Args:
        w_normal (ndarray, 3HWS): per-sample world-space positions
        pv (ndarray, size (4,4)): camera view-projection matrix
        height (int): height of the camera resolution (in pixels)
        width (int): width of the camera resolution (in pixels)

    Returns:
        projected (ndarray, 2HWS): Per-sample screen-space position (pixel coordinates).
            IJ INDEXING! for gather ops and consistency,
            see backproject_pixel_centers in noisebase.torch.projective for use with grid_sample.
            Degenerate positions give inf.
    """
    # TODO: support any number of extra dimensions like apply_array
    homogeneous = np.concatenate((  # Pad to homogeneous coordinates
        w_position,
        np.ones_like(w_position)[0:1]
    ))

    # ROW VECTOR ALERT!
    # DirectX uses row vectors...
    projected = np.einsum('ij, ihws -> jhws', pv, homogeneous)
    projected = np.divide(
        projected[0:2], projected[3],
        out=np.zeros_like(projected[0:2]),
        where=projected[3] != 0
    )

    # directx pixel coordinate fluff
    projected = projected * np.reshape([0.5 * width, -0.5 * height], (2, 1, 1, 1)).astype(np.float32) \
                + np.reshape([width / 2, height / 2], (2, 1, 1, 1)).astype(np.float32)

    projected = np.flip(projected, 0)  # height, width; ij indexing

    return projected
def motion_vectors(w_position, w_motion, pv, prev_pv, height, width):
    """Computes per-sample screen-space motion vectors (in pixels)

    Args:
        w_position (ndarray, 3HWS): per-sample world-space positions
        w_motion (ndarray, 3HWS): per-sample world-space positions
        pv (ndarray, size (4,4)): camera view-projection matrix
        prev_pv (ndarray, size (4,4)): camera view-projection matrix from previous frame
        height (int): height of the camera resolution (in pixels)
        width (int): width of the camera resolution (in pixels)

    Returns:
        motion (ndarray, 2HWS): Per-sample screen-space motion vectors (in pixels).
            IJ INDEXING! for gather ops and consistency,
            see backproject_pixel_centers in noisebase.torch.projective for use with grid_sample.
            Degenerate positions give inf.
    """
    # TODO: support any number of extra dimensions like apply_array (only the docstring here)
    current = screen_space_position(w_position, pv, height, width)
    prev = screen_space_position(w_position + w_motion, prev_pv, height, width)

    motion = prev - current

    return motion
def decompress_RGBE(color, exposures):
    """Decompresses per-sample radiance from RGBE compressed data

    Args:
        color (ndarray, uint8, 4HWS): radiance data in RGBE representation
        [min_exposure, max_exposure]: exposure range for decompression

    Returns:
        color (ndarray, 3HWS): per-sample RGB radiance
    """
    exponents = (color.astype(np.float32)[3] + 1)/256
    #exposures = np.reshape(exposures, (1, 1, 1, 2))

    exponents = np.exp(exponents * (exposures[1] - exposures[0]) + exposures[0])
    color = color.astype(np.float32)[:3] / 255 * exponents[np.newaxis]
    return color
def numtostring(j):
    l11 = (j // 1000) % 10
    l12 = (j // 100) % 10
    l13 = (j // 10) % 10
    l14 = j % 10
    order = str(l11) + str(l12) + str(l13) + str(l14)
    current = "frame" + order + ".zip"
    return current

pathname = ["bistro1"]  # 修改为你的场景名称
samplec = 0
end = "./data/"  # 修改为你的数据路径
for dirs in pathname:
    endpath = end + dirs + "/"
    for i in range(160):
        partpath = numtostring(i)
        curpath = dirs + "/"
        zippath = curpath + partpath
        ds = zarr.open_group(store = ZipStore(end+zippath, mode='r'),mode='r')
        
        color_data = np.array(ds['color'])
        exposure_data = np.array(ds['exposure'])
        # 启用需要验证的数据通道
        color = decompress_RGBE(color_data, exposure_data)
        # albedo = np.array(ds['diffuse'])
        normal = np.array(ds['normal'])
        # motion = np.array(ds['motion'])
        position = np.array(ds['position'])
        reference = np.array(ds['reference'])  # 启用reference
        # camera_position = np.array(ds.camera_position)
        # mat = np.array(ds.view_proj_mat)

        # if i == 0:
        #     motion2 = projective.motion_vectors(position, motion, mat, mat, position.shape[1], position.shape[2])
        # else:
        #     motion2 = projective.motion_vectors(position, motion, mat, pvmat, position.shape[1], position.shape[2])

        # pvmat = mat
        #
        # tmotion = motion2.mean(axis=3)

        # tmotion = np.transpose(tmotion, (1, 2, 0))
        #
        # single = np.zeros((tmotion.shape[0],tmotion.shape[1],1))
        # tmotion = np.concatenate((tmotion, single),axis= 2)

        # depth = projective.log_depth(position, camera_position)

        # 处理各种数据通道
        # t_depth = depth.mean(axis=3)
        tcolor = color.mean(axis=3)
        # talbedo = albedo.mean(axis=3)
        tnormal = normal.mean(axis=3)
        tposition = position.mean(axis=3)
        ref = np.transpose(reference, (1, 2, 0))  # reference直接转置


        # t_depth = np.transpose(t_depth, (1, 2, 0))
        tcolor = np.transpose(tcolor, (1, 2, 0))
        # talbedo = np.transpose(talbedo, (1, 2, 0))
        tnormal = np.transpose(tnormal, (1, 2, 0))
        tposition = np.transpose(tposition, (1, 2, 0))

        # 创建输出目录并保存（只处理前5帧，避免生成太多文件）
        if i < 5:
            import os
            os.makedirs("./detach_bistro1/reference", exist_ok=True)
            os.makedirs("./detach_bistro1/color", exist_ok=True)
            os.makedirs("./detach_bistro1/normal", exist_ok=True)
            os.makedirs("./detach_bistro1/position", exist_ok=True)
            
            write_RGB_exr("./detach_bistro1/reference/ref-" + str(i) + ".exr", ref)
            write_RGB_exr("./detach_bistro1/color/color-" + str(i) + ".exr", tcolor)
            write_RGB_exr("./detach_bistro1/normal/normal-" + str(i) + ".exr", tnormal)
            write_RGB_exr("./detach_bistro1/position/position-" + str(i) + ".exr", tposition)
            
            print(f"已处理帧 {i}")
        
        # 只处理前5帧就停止
        if i >= 4:
            break

    samplec = samplec + 1

