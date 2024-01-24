import numpy as np
from PIL import Image

def range_projection(current_vertex, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50):
    """ Project a pointcloud into a spherical projection, range image.
    Args:
        current_vertex: raw point clouds
    Returns:
        proj_range: projected range image with depth, each pixel contains the corresponding depth
        proj_vertex: each pixel contains the corresponding point (x, y, z, 1)
        proj_intensity: each pixel contains the corresponding intensity
        proj_idx: each pixel contains the corresponding index of the point in the raw point cloud
    """
    # laser parameters
    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

    # get depth of all points
    depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
    current_vertex = current_vertex[(depth > 0) & (depth < max_range)]  # get rid of [0, 0, 0] points
    depth = depth[(depth > 0) & (depth < max_range)]

    # get scan components
    scan_x = current_vertex[:, 0]
    scan_y = current_vertex[:, 1]
    scan_z = current_vertex[:, 2]
    intensity = current_vertex[:, 3]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    intensity = intensity[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    scan_x = scan_x[order]
    scan_y = scan_y[order]
    scan_z = scan_z[order]

    indices = np.arange(depth.shape[0])
    indices = indices[order]

    proj_range = np.full((proj_H, proj_W), -1,
                        dtype=np.float32)  # [H,W] range (-1 is no data)
    proj_vertex = np.full((proj_H, proj_W, 4), -1,
                        dtype=np.float32)  # [H,W] index (-1 is no data)
    proj_idx = np.full((proj_H, proj_W), -1,
                        dtype=np.int32)  # [H,W] index (-1 is no data)
    proj_intensity = np.full((proj_H, proj_W), -1,
                            dtype=np.float32)  # [H,W] index (-1 is no data)

    proj_range[proj_y, proj_x] = depth
    proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z, np.ones(len(scan_x))]).T
    proj_idx[proj_y, proj_x] = indices
    proj_intensity[proj_y, proj_x] = intensity
    # tensor
    # circular convolutions? remove overlap?
    return proj_range, proj_vertex, proj_intensity, proj_idx

def loadCloudFromBinary(file, cols=3):
    bin_pcd = np.fromfile(file, dtype=np.float32)
    points = bin_pcd.reshape(-1, 4)
    return points
    
def createRangeImage(points, crop=True, crop_distance=False, distance_threshold=100):
    if crop_distance:
        dis = np.linalg.norm(points[:, :3], axis=1)
        points = points[np.where(dis<distance_threshold)[0]]
    image,_,_,_ = range_projection(points)
    image = Image.fromarray(np.uint8(image * 5.1) , 'L')
    image = np.stack((image,)*3, axis=-1)
    if crop:
        clipStart = image.shape[1] // 3      
        image = image[0:32, clipStart:clipStart*2, :]
    return image

def loadLidarImage(filename, seq, data_path):
    lidar_points = loadCloudFromBinary(f"{data_path}/{seq}/velodyne/{filename}.bin")
    lidar_image = createRangeImage(lidar_points)
    return lidar_image
