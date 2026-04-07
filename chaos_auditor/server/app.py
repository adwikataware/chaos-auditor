from openenv.core.env_server import create_fastapi_app

from chaos_auditor.models import ChaosAction, SystemObservation
from chaos_auditor.server.environment import ChaosAuditorEnvironment

app = create_fastapi_app(ChaosAuditorEnvironment, ChaosAction, SystemObservation)
