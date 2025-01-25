from .geotiff_dataset import GeoTIFFDataset
import numpy as np
import rasterio.io
import rioxarray as rxr
import xarray
import pytest


@pytest.mark.parametrize(
    "raster_file_path",
    [
        "one_band_raster_file_path",
        "three_band_raster_file_path",
    ],
)
class TestUnitGeoTIFFDataset:
    @staticmethod
    def test_raster_file_correctly_mocked(request, raster_file_path):
        raster_file_path = request.getfixturevalue(raster_file_path)

        with rasterio.open(raster_file_path) as dataset:
            profile = dataset.profile.data
            array = dataset.read()

        assert array.shape == (profile["count"], profile["height"], profile["width"])
        assert np.all(array == 1)

    @staticmethod
    def test_geotiffdataset_load(request, raster_file_path):
        raster_file_path = request.getfixturevalue(raster_file_path)

        with rasterio.open(raster_file_path) as dataset:
            reference_array = dataset.read()
            reference_profile = dataset.profile.data

        dataset = GeoTIFFDataset(filepath=raster_file_path)
        assert isinstance(dataset.load(), xarray.DataArray)

        raster = dataset.load()
        assert raster.shape == reference_array.shape
        assert raster.dtype == reference_array.dtype

        assert raster.rio.transform() == reference_profile["transform"]
        assert raster.rio.crs == reference_profile["crs"]

    @staticmethod
    def test_geotiffdataset_save(
        request, raster_file_path, data_array_of_ones_from_shape
    ):
        raster_file_path = request.getfixturevalue(raster_file_path)
        with rasterio.open(raster_file_path) as dataset:
            reference_profile = dataset.profile.data

        data_array = data_array_of_ones_from_shape(
            (
                reference_profile["count"],
                reference_profile["height"],
                reference_profile["width"],
            )
        )

        with rasterio.io.MemoryFile() as memfile:
            GeoTIFFDataset(filepath=memfile.name).save(data_array)

            data = GeoTIFFDataset(filepath=memfile.name).load()
            assert data.shape == data_array.shape
            assert data.dtype == data_array.dtype
            assert data.rio.transform() == data_array.rio.transform()
            assert data.rio.crs == data_array.rio.crs
