import pytest

from matchzoo.engine.base_model import BaseModel


def test_base_model_abstract_instantiation():
    with pytest.raises(TypeError):
        model = BaseModel(BaseModel.get_default_params())
        assert model


def test_base_model_concrete_instantiation():
    class MyBaseModel(BaseModel):
        def build(self):
            self.a, self.b = 1, 2
        def forward(self):
            return self.a + self.b

    model = MyBaseModel()
    assert model.params
    model.guess_and_fill_missing_params()
    model.build()
    assert model.params.completed(exclude=['out_activation_func'])
