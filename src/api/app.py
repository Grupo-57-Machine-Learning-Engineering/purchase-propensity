"""API de previsão de propensão de compra. Rode com `make api`.

Carrega o `champion` do Model Registry ao subir e expõe:

- `POST /predict`: score de um lote de visitas;
- `GET /health`: estado da API e a versão do modelo carregado;
- `GET /docs`: documentação interativa (Swagger), para testar pelo navegador.
"""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse

from src import __version__
from src.api.model import LoadedModel, load_champion, predict
from src.api.schemas import Health, PredictRequest, PredictResponse
from src.config import DECISION_THRESHOLD
from src.logger import get_logger

ModelLoader = Callable[[], LoadedModel]

DESCRIPTION = (
    "Ordena visitas de uma loja online pela chance de pedido, a partir das flags de "
    "navegação da sessão. O score serve para priorizar (por exemplo, quem recebe uma "
    "oferta), não é probabilidade calibrada. Detalhes em `docs/model_card.md`."
)

logger = get_logger(__name__)


def _loaded(request: Request) -> LoadedModel:
    return request.app.state.model


def create_app(loader: ModelLoader = load_champion) -> FastAPI:
    """Monta a API. `loader` carrega o modelo ao subir (os testes passam um falso)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.model = loader()
        logger.info("Modelo carregado: %s versão %s", app.state.model.name, app.state.model.version)
        yield

    app = FastAPI(
        title="purchase-propensity", version=__version__, description=DESCRIPTION, lifespan=lifespan
    )

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.get("/health")
    def health(request: Request) -> Health:
        loaded = _loaded(request)
        return Health(status="ok", model_name=loaded.name, model_version=loaded.version)

    @app.post("/predict")
    def predict_visitors(body: PredictRequest, request: Request) -> PredictResponse:
        loaded = _loaded(request)
        return PredictResponse(
            model_name=loaded.name,
            model_version=loaded.version,
            threshold=DECISION_THRESHOLD,
            predictions=predict(loaded, body.visitors, DECISION_THRESHOLD),
        )

    return app


app = create_app()
