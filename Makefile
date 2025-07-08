RESULT_DIR=?
DASHBOARD_DIR=?


format:
	ruff format move_to_dashboard.py
	isort move_to_dashboard.py

move:
	python move_to_dashboard.py \
	  --trainer_dir $(RESULT_DIR) \
	  --dashboard_dir $(DASHBOARD_DIR)