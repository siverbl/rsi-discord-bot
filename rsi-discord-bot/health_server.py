# health_server.py
from aiohttp import web
import logging

logger = logging.getLogger(__name__)

async def _health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok"})

async def start_health_server(host: str = "127.0.0.1", port: int = 8080) -> web.AppRunner:
    """
    Local-only server: http://127.0.0.1:<port>/health
    """
    app = web.Application()
    app.router.add_get("/health", _health)

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, host=host, port=port)
    await site.start()

    logger.info(f"Health server running on http://{host}:{port}/health")
    return runner
