import ez_torch as ez
from ez_torch.models import SpatialUVOffsetTransformer


def test_version():
    assert ez.__version__ == "0.1.0"


def test_SpatialUVOffsetTransformer():
    model = SpatialUVOffsetTransformer(1000, uv_resolution_shape=(10, 10))
