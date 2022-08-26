import numpy as np
import pyroomacoustics as pra
from tqdm import tqdm

def generate_room_dims(vol, var=.3):
    cube_length = np.power(vol, 1./3)
    side1, side2 = np.random.normal(loc=cube_length, scale=var*cube_length, size=2)
    side3 = vol/(side1*side2)
    return np.array([side1, side2, side3])

def generate_in_room(room_center, room_dims, loc_std, lim=100):
    loc = np.random.multivariate_normal(room_center, np.identity(3)*loc_std)
    count = 0
    while np.any(loc>room_dims) or np.any(loc<0):
        loc = np.random.multivariate_normal(room_center, np.identity(3)*loc_std)
        count += 1
        if count>lim:
            print("Limit {} reached, returning room center")
            loc = room_dims/2
            break
    return loc

def generate_rirs(room_dims, n_mics, materials, sr=44100, max_order=10, loc_std=None):
    room_center = room_dims/2
    if loc_std is None:
        loc_std = np.average(room_dims)/3
    
    # Create the room
    room = pra.ShoeBox(
       room_dims, fs=sr, materials=materials, max_order=max_order,
       air_absorption=True, ray_tracing=True
    )
    
    source_loc = generate_in_room(room_center, room_dims, loc_std)
    try:
        room.add_source(source_loc)
    except Exception:
        print("{} in {} ".format(source_loc, room_dims))
    
    mic_locs = []
    for _ in range(n_mics):
        mic_locs.append(generate_in_room(room_center, room_dims, loc_std))
    room.add_microphone(np.array(mic_locs).T)

    room.compute_rir()
    rirs_out = []
    for i in range(n_mics):
        rirs_out.append(room.rir[i][0])
    return rirs_out