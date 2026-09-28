import json
import os
import re
import threading
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query


ROOT_DIR = Path(__file__).resolve().parent
FORECAST_PATH = Path(
	os.getenv(
		"SELLWISE_FORECAST_PATH",
		str(ROOT_DIR / "artifacts/model_evaluation/submissions/submission_final.csv"),
	)
)
CALENDAR_PATH = Path(
	os.getenv(
		"SELLWISE_CALENDAR_PATH",
		str(ROOT_DIR / "artifacts/data_ingestion/dataset/calendar.csv"),
	)
)
METRICS_PATH = Path(
	os.getenv(
		"SELLWISE_METRICS_PATH",
		str(ROOT_DIR / "artifacts/model_evaluation/metrics.json"),
	)
)

FORECAST_COLUMNS = [f"F{day}" for day in range(1, 29)]
ITEM_ID_PATTERN = re.compile(
	r"^(?P<item_id>.+)_(?P<state_id>CA|TX|WI)_(?P<store_number>[1-4])_evaluation$"
)


class ForecastRepository:
	def __init__(self) -> None:
		self._lock = threading.Lock()
		self._frame: pd.DataFrame | None = None
		self._dates: list[str] | None = None

	@property
	def available(self) -> bool:
		return FORECAST_PATH.is_file() and CALENDAR_PATH.is_file()

	def load(self) -> tuple[pd.DataFrame, list[str]]:
		if self._frame is not None and self._dates is not None:
			return self._frame, self._dates

		with self._lock:
			if self._frame is not None and self._dates is not None:
				return self._frame, self._dates
			if not FORECAST_PATH.is_file():
				raise HTTPException(
					status_code=503,
					detail=f"Forecast artifact not found: {FORECAST_PATH}",
				)
			if not CALENDAR_PATH.is_file():
				raise HTTPException(
					status_code=503,
					detail=f"Calendar artifact not found: {CALENDAR_PATH}",
				)

			forecast = pd.read_csv(FORECAST_PATH)
			missing_columns = [
				column
				for column in ["id", *FORECAST_COLUMNS]
				if column not in forecast.columns
			]
			if missing_columns:
				raise HTTPException(
					status_code=503,
					detail=f"Forecast artifact is missing columns: {missing_columns}",
				)

			metadata = forecast["id"].astype(str).str.extract(ITEM_ID_PATTERN)
			if metadata.isna().any().any():
				raise HTTPException(
					status_code=503,
					detail="Forecast IDs must end in _CA_1_evaluation, _TX_1_evaluation, or _WI_1_evaluation.",
				)

			forecast["item_id"] = metadata["item_id"]
			forecast["state_id"] = metadata["state_id"]
			forecast["store_id"] = (
				metadata["state_id"] + "_" + metadata["store_number"]
			)
			item_parts = forecast["item_id"].str.split("_")
			forecast["department_id"] = item_parts.str[:2].str.join("_")

			calendar = pd.read_csv(CALENDAR_PATH, usecols=["d", "date"])
			calendar["day_number"] = calendar["d"].str.extract(r"d_(\d+)")[0].astype(int)
			calendar = calendar.sort_values("day_number").tail(len(FORECAST_COLUMNS))
			if len(calendar) != len(FORECAST_COLUMNS):
				raise HTTPException(
					status_code=503,
					detail="Calendar does not contain a complete 28-day forecast horizon.",
				)

			self._frame = forecast
			self._dates = calendar["date"].astype(str).tolist()
			return self._frame, self._dates


repository = ForecastRepository()
app = FastAPI(
	title="SellWise Forecast API",
	description="Read-only API for the generated SellWise forecast and evaluation metrics.",
	version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, object]:
	return {
		"status": "ok",
		"forecast_available": repository.available,
		"metrics_available": METRICS_PATH.is_file(),
	}


@app.get("/horizon")
def horizon() -> dict[str, object]:
	_, dates = repository.load()
	return {"start_date": dates[0], "end_date": dates[-1], "dates": dates}


@app.get("/catalog")
def catalog(
	store_id: str | None = None,
	department_id: str | None = None,
) -> dict[str, list[str]]:
	forecast, _ = repository.load()
	filtered = forecast
	if store_id:
		filtered = filtered[filtered["store_id"] == store_id]
	if department_id:
		filtered = filtered[filtered["department_id"] == department_id]

	return {
		"stores": sorted(forecast["store_id"].unique().tolist()),
		"departments": sorted(filtered["department_id"].unique().tolist()),
		"items": (
			sorted(filtered["item_id"].unique().tolist()) if store_id else []
		),
	}


@app.get("/forecasts")
def forecasts(
	store_id: str,
	item_id: str | None = None,
	department_id: str | None = None,
	start_date: date | None = None,
	end_date: date | None = None,
	limit: int = Query(default=100, ge=1, le=500),
	offset: int = Query(default=0, ge=0),
) -> dict[str, object]:
	if start_date and end_date and start_date > end_date:
		raise HTTPException(
			status_code=422,
			detail="start_date must be on or before end_date.",
		)

	forecast, dates = repository.load()
	filtered = forecast[forecast["store_id"] == store_id]
	if item_id:
		filtered = filtered[filtered["item_id"] == item_id]
	if department_id:
		filtered = filtered[filtered["department_id"] == department_id]

	total_series = len(filtered)
	selected = filtered.sort_values("item_id").iloc[offset : offset + limit]
	selected_dates = [
		(index, forecast_date)
		for index, forecast_date in enumerate(dates)
		if (start_date is None or date.fromisoformat(forecast_date) >= start_date)
		and (end_date is None or date.fromisoformat(forecast_date) <= end_date)
	]

	data: list[dict[str, object]] = []
	for row in selected.itertuples(index=False):
		row_values = row._asdict()
		for day_index, forecast_date in selected_dates:
			data.append(
				{
					"id": row_values["id"],
					"item_id": row_values["item_id"],
					"store_id": row_values["store_id"],
					"department_id": row_values["department_id"],
					"date": forecast_date,
					"forecast": float(row_values[FORECAST_COLUMNS[day_index]]),
				}
			)

	return {
		"store_id": store_id,
		"total_series": total_series,
		"limit": limit,
		"offset": offset,
		"data": data,
	}


@app.get("/metrics")
def metrics() -> dict[str, float]:
	if not METRICS_PATH.is_file():
		raise HTTPException(
			status_code=503,
			detail=f"Metrics artifact not found: {METRICS_PATH}",
		)
	try:
		with METRICS_PATH.open("r", encoding="utf-8") as metrics_file:
			return json.load(metrics_file)
	except (OSError, json.JSONDecodeError) as error:
		raise HTTPException(
			status_code=503,
			detail=f"Could not read metrics artifact: {error}",
		) from error
