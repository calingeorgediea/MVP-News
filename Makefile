# Simple Makefile for Python

INPUT_CSV = ./data/raw/categories/actualitate.csv
OUTPUT_DIR = ./data/processed/categories/actualitate.csv

.PHONY: run test install clean

# Define the default target
default: getdata

# Target to run the Python script
getdata:
	python ./src/data/get_data.py

move_csv:
	mv mongodb_data.csv ./data/raw/

separate_data:
	python ./src/data/separate_datas.py

process_data:
	python ./src/data/app_process_data.py ./data/raw/categories ./data/interim/categories

find_topics:
	python ./src/data/find_topics.py ./data/raw/categories ./data/interim/categories

process_files:
	@echo "Processing files in ./data/interim/categories/..."
	@for file in ./data/interim/categories/*; do \
		if [ -f "$$file" ]; then \
			echo "Processing $$file..."; \
			python your_script.py "$$file"; \
		fi \
	done
	@echo "Processing complete."


normalize_results_from_spark:
	@echo "Normalizing results from Spark..."
	@for dir in ./data/processed/categories/*; do \
		if [ -d "$$dir" ]; then \
			echo "Processing $$dir..."; \
			output_file="$$dir/merged.csv"; \
			python ./src/data/normalize_data_structure.py "$$dir" "./data/processed/news"; \
		fi \
	done
	@echo "Normalization complete."

train_lda:
	@echo "Processing files in /data/processed/news/..."
	@for file in ./data/processed/news/*; do \
		if [ -f "$$file" ]; then \
			echo "Processing $$file..."; \
			python models/app_model_01.py "$$file"; \
		fi \
	done
	@echo "Processing complete."


# Target to run tests (if any)
test:
	# Add commands here to run tests
	# For example, you could use pytest
	# pytest tests/

# Target to install dependencies (if any)
install:
	# Add commands here to install dependencies
	# For example, using pip
	# pip install -r requirements.txt

# Target to clean up temporary files (if any)
clean:
	# Add commands here to clea
