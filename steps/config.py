from pydantic import BaseModel, ConfigDict

class ModelNameConfig(BaseModel):
    model_name: str = "linear_regression"
    fine_tuning: bool = False
    model_config = ConfigDict(protected_namespaces=())
ModelNameConfig = ModelNameConfig()
