import asyncio
from io import BytesIO

from httpx import ASGITransport, AsyncClient

from app import app


def run_async(coro):
    return asyncio.run(coro)


def test_health_endpoint_returns_status():
    async def _run():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body
        assert "checks" in body
        assert "model_version" in body

    run_async(_run())


def test_compare_requires_two_files_minimum():
    async def _run():
        one_csv = BytesIO(b"a,b\n1,2\n3,4\n5,6\n7,8\n9,10\n")
        files = {"files": ("single.csv", one_csv, "text/csv")}
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/compare", files=files)
        assert resp.status_code == 400
        assert "between 2 and 10" in resp.json()["detail"]

    run_async(_run())


def test_predict_rejects_non_csv_file():
    async def _run():
        bad_file = BytesIO(b"not,csv,for,predict")
        files = {"file": ("bad.txt", bad_file, "text/plain")}
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/predict", files=files)
        assert resp.status_code == 400
        assert "only .csv files are allowed" in resp.json()["detail"]

    run_async(_run())


def test_metrics_endpoint_exposes_counters():
    async def _run():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            _ = await client.get("/health")
            resp = await client.get("/metrics")
        assert resp.status_code == 200
        body = resp.json()
        assert "requests_total" in body
        assert "requests_by_path" in body
        assert "latency_ms_by_path" in body
        assert body["requests_total"] >= 1

    run_async(_run())
