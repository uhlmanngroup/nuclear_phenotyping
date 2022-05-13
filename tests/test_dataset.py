import pytest
from cellesce import Cellesce

class TestData:
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.data_folder = "analysed/_2019_cellesce_unet_splineparameters_aligned/raw/projection_XY/"
        self.data = Cellesce(data_folder=self.data_folder)

    def data_construct_test(self):
        return Cellesce(data_folder=self.data_folder)
    
    def test_data_construct_test(self):
        return self.data_construct_test()
    
    def test_one(self):
            self.value = 1
            assert self.value == 1
# %%

# Cellesce("analysed/_2019_cellesce_unet_splineparameters_aligned/raw/projection_XY/")
# %%
