from scipy.spatial.transform import Rotation as R


def matrix_to_quaternion(rot_matrix):
    r = R.from_matrix(rot_matrix)
    quat = r.as_quat()
    return quat