.PHONY: up down train stream dashboard test lint clean help

help:
	@echo "Usage:"
	@echo "  make up         Start all infrastructure (Kafka, Redis, Postgres, etc.)"
	@echo "  make down       Stop all infrastructure"
	@echo "  make train      Train XGBoost + LightGBM models"
	@echo "  make stream     Start Kafka producer + consumer"
	@echo "  make dashboard  Launch Streamlit dashboard"
	@echo "  make test       Run test suite"
	@echo "  make lint       Run flake8 + isort"
	@echo "  make clean      Remove artifacts and __pycache__"

up:
	docker-compose up -d
	@echo "Waiting for services to be healthy..."
	@sleep 15
	@echo "Services ready. MLflow: http://localhost:5000 | Grafana: http://localhost:3000 | Dashboard: http://localhost:8501"

down:
	docker-compose down

train:
	@echo "Training XGBoost champion..."
	python src/models/train_xgb.py
	@echo "Training LightGBM challenger..."
	python src/models/train_lgbm.py
	@echo "Models saved to models/"

stream:
	@echo "Starting producer + consumer..."
	python data/synthetic/generate_stream.py &
	python src/streaming/consumer.py &
	@echo "Streaming pipeline live. Check dashboard at http://localhost:8501"

dashboard:
	streamlit run app/streamlit_dashboard.py --server.port 8501

test:
	pytest tests/ -v --tb=short

lint:
	flake8 src/ app/ tests/ --max-line-length=120
	isort src/ app/ tests/ --check-only

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage
	rm -rf outputs/drift_report.html
