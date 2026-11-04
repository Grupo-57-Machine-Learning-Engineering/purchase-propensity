"""Contrato da API: entrada com as 20 flags de navegação e saída com o score.

Os campos de `Visitor` são as colunas que o modelo vê no treino, na mesma ordem
(`src/features/selection.py`): as 23 flags do dataset menos as 3 de vazamento
(`LEAKAGE_COLUMNS`). Ao subir, a API confere esses nomes contra os do modelo
carregado (`src/api/model.py`).
"""

from pydantic import BaseModel, ConfigDict, Field

MAX_VISITORS = 1000
USER_ID_FIELD = "user_id"


class Visitor(BaseModel):
    """Uma visita à loja: o que a pessoa fez durante a sessão."""

    model_config = ConfigDict(extra="forbid")

    user_id: str | None = Field(
        default=None, description="Identificador opcional, devolvido na resposta. Não é feature."
    )
    basket_icon_click: bool
    basket_add_list: bool
    basket_add_detail: bool
    sort_by: bool
    image_picker: bool
    account_page_click: bool
    promo_banner_click: bool
    detail_wishlist_add: bool
    list_size_dropdown: bool
    closed_minibasket_click: bool
    checked_returns_detail: bool
    saw_sizecharts: bool
    saw_delivery: bool
    saw_account_upgrade: bool
    saw_homepage: bool
    device_mobile: bool
    device_computer: bool
    device_tablet: bool
    returning_user: bool
    loc_uk: bool


FEATURE_COLUMNS: list[str] = [name for name in Visitor.model_fields if name != USER_ID_FIELD]

EXAMPLE_VISITOR: dict[str, str | bool] = {
    USER_ID_FIELD: "visita-1",
    **dict.fromkeys(FEATURE_COLUMNS, False),
    "basket_icon_click": True,
    "basket_add_detail": True,
    "saw_homepage": True,
    "device_computer": True,
    "returning_user": True,
    "loc_uk": True,
}


class PredictRequest(BaseModel):
    """Lote de visitas para pontuar (até `MAX_VISITORS` por chamada)."""

    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"examples": [{"visitors": [EXAMPLE_VISITOR]}]}
    )

    visitors: list[Visitor] = Field(min_length=1, max_length=MAX_VISITORS)


class Prediction(BaseModel):
    """Score de uma visita."""

    user_id: str | None
    score: float = Field(
        description="Score do modelo, de 0 a 1. Serve para ordenar as visitas; não é "
        "probabilidade calibrada (o peso de classe balanceado infla os valores)."
    )
    purchase: bool = Field(description="`score >= threshold`.")


class PredictResponse(BaseModel):
    """Scores na mesma ordem das visitas recebidas, e o modelo que os gerou."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    threshold: float
    predictions: list[Prediction]


class Health(BaseModel):
    """Estado da API e o modelo carregado."""

    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_name: str
    model_version: str
