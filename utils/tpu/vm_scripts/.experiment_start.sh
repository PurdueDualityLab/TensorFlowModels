zone="$(gcloud compute instances list --filter="name ~ \^$HOSTNAME\$" --format 'csv[no-heading](zone)')"
gcloud compute tpus start $1 --zone=$zone
trap gcloud compute tpus stop $1 --zone=$zone EXIT
echo $zone
TPU_NAME=$1 python3 -m $2.train_vm --mode=train_and_eval --experiment=$3 --config_file=$2/configs/experiments/$4.yaml --model_dir=gs://tensorflow2/$5
