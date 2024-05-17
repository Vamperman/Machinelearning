import h5py


# Open the HDF5 file
with h5py.File('output/test1/model.h5', 'r') as file:
    # Explore the contents
    print(list(file.keys()))
