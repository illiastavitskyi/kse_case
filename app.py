from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

import service


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.start()   # перший розрахунок + щогодинне оновлення у фоновому потоці
    yield


app = FastAPI(title="Прогноз повітряних тривог", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/alerts")


@app.get("/alerts")
def alerts(request: Request):
    return templates.TemplateResponse(request, "alerts.html", {"f": service.get_forecast()})
