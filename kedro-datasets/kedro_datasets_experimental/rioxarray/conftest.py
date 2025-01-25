import pytest
import rasterio.io
import numpy as np
import xarray as xr


@pytest.fixture(scope="session")
def data_array_of_ones_from_shape():
    def data_array_of_ones_from_shape(shape: tuple[int, int, int]) -> xr.DataArray:
        data = xr.DataArray(
            data=np.ones(shape, dtype="float32"),
            dims=("band", "y", "x"),
            coords={
                "band": np.arange(1, shape[0] + 1),
                "y": np.arange(shape[1]),
                "x": np.arange(shape[2]),
            },
        )

        data.rio.write_crs("epsg:4326", inplace=True)
        data.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

        data = data.assign_attrs(
            {f"band_{i}_description": f"band_{i}" for i in range(1, shape[0] + 1)}
        )
        return data

    return data_array_of_ones_from_shape


@pytest.fixture(scope="session")
def one_band_raster_file_path(data_array_of_ones_from_shape):
    memfile = rasterio.io.MemoryFile()
    data_array_of_ones_from_shape((1, 10, 10)).rio.to_raster(memfile.name)
    yield memfile.name
    memfile.close()


@pytest.fixture(scope="session")
def three_band_raster_file_path(data_array_of_ones_from_shape):
    memfile = rasterio.io.MemoryFile()
    data_array_of_ones_from_shape((3, 10, 10)).rio.to_raster(memfile.name)
    yield memfile.name
    memfile.close()
