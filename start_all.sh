#!/usr/bin/env bash

nohup ./_start_all.sh > test.log &
tail -f test.log