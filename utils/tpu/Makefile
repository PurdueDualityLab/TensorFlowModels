ifeq (, $(shell which ctpu))
$(error "No ctpu in $(PATH), consider downlading Google Cloud SDK (https://cloud.google.com/sdk/docs/install) and CTPU (https://github.com/tensorflow/tpu/tree/master/tools/ctpu)")
endif

.PHONY: tpu tpu_prior_settings

tpu: creds/default.txt
	[ ! -e .config ] || rm .config
	python3 show_menu.py

tpu_prior_settings: creds/default.txt
	python3 show_menu.py

creds/default.txt:
	python3 create_kconfig.py
