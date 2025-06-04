import os


feature_all_dir = "/home/fi5666wi/PRIAS_data/features_uni_v2_all"
feature_gg_dir = "/home/fi5666wi/PRIAS_data/features_from_bengio"


def modify_h5_files(feature_all_dir, feature_gg_dir):
    import h5py
    import numpy as np

    for filename in os.listdir(feature_all_dir):
        if filename.endswith(".h5"):
            all_path = os.path.join(feature_all_dir, filename)
            gg_path = os.path.join(feature_gg_dir, filename)

            with h5py.File(gg_path, 'r+') as gg_file:
                with h5py.File(all_path, 'r+') as all_file:
                    gg_coords = gg_file['coords'][:]
                    all_coords = all_file['coords'][:]
                    # Check if the coordinates match
                    row_to_index = {tuple(row): i for i, row in enumerate(gg_coords)}
                    order = [row_to_index[tuple(row)] for row in all_coords]
                    if np.array_equal(gg_coords[order], all_coords):
                        if 'gg_scores' in all_file:
                            del all_file['gg_scores']
                        all_file.create_dataset('gg_scores', data=gg_file['gg_scores'][:][order], dtype=gg_file['gg_scores'].dtype)
                        #all_file['gg_scores'][:] = gg_file['gg_scores'][order]
                        print(f"Updated {filename} with gg scores.")
                    else:
                        print(f"Coordinates do not match for {filename}. Skipping update.")

if __name__ == "__main__":
    modify_h5_files(feature_all_dir, feature_gg_dir)
    print("Modification of h5 files completed.")