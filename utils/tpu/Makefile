ifeq (, $(shell which ctpu))
$(error "No ctpu in $(PATH), consider downlading Google Cloud SDK (https://cloud.google.com/sdk/docs/install) and CTPU (https://github.com/tensorflow/tpu/tree/master/tools/ctpu)")
endif

.PHONY: new_tpu menuconfig

new: Kconfig
	python3 show_menu.py

brand_new: Kconfig
	[ ! -e .config ] || rm .config
	python3 show_menu.py

Kconfig: Kconfig.template creds/*.json
	python3 create_kconfig.py
