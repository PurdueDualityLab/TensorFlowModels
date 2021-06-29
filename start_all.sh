#!/usr/bin/env bash

chmod +x _start_all.sh 
nohup ./_start_all.sh $1 > run.log &
tail -f run.log