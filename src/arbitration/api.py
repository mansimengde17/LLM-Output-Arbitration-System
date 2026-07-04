"""FastAPI service for the arbitration system."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .graph import arbitrate
from .store import VerdictStore

app = FastAPI(title="LLM Output Arbitration System", version="1.0.0")
store = VerdictStore()


class ArbitrateRequest(BaseModel):
    output: str
    question: str = ""


class BatchRequest(BaseModel):
    items: list[ArbitrateRequest]


@app.post("/v1/arbitrate")
async def arbitrate_one(request: ArbitrateRequest):
    arbitration = await arbitrate(request.output, request.question)
    store.save(arbitration)
    return arbitration


@app.post("/v1/arbitrate/batch")
async def arbitrate_batch(request: BatchRequest):
    results = []
    for item in request.items:
        arbitration = await arbitrate(item.output, item.question)
        store.save(arbitration)
        results.append({
            "arbitration_id": arbitration.arbitration_id,
            "excerpt": item.output[:80],
            "overall_score": arbitration.verdict.overall_score,
            "issues": len(arbitration.verdict.confirmed_issues),
            "confidence": arbitration.verdict.confidence,
        })
    return {"results": sorted(results, key=lambda r: r["overall_score"])}


@app.get("/v1/arbitrations/{arbitration_id}")
def get_arbitration(arbitration_id: str):
    record = store.load(arbitration_id)
    if record is None:
        raise HTTPException(404, "arbitration not found")
    return record


@app.get("/v1/analytics")
def analytics():
    return store.analytics()
