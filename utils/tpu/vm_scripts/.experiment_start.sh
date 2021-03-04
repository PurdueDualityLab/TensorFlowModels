if [ -f ".experiment_zone" ]
then
  gcloud compute instances list --filter="name ~ ^$HOSTNAME\$" --format 'csv[no-heading](zone)' > .experiment_zone
fi
zone="$(cat .experiment_zone)"
zone=europe-west4-a
tpu=$1
gcloud compute tpus start $1 --zone=$zone
stopvm() {
        gcloud compute tpus stop $tpu --zone=$zone
}
trap stopvm EXIT
echo $zone
if [ $# -ge 5 ]
then
  TPU_NAME=$1 python3 -m $2.train_vm --mode=train_and_eval --experiment=$3 --config_file=$2/configs/experiments/$4.yaml --model_dir=gs://tensorflow2/$4-$5
else
  TPU_NAME=$1 python3 -m $2.train_vm --mode=train_and_eval --experiment=$3 --config_file=$2/configs/experiments/$4.yaml --model_dir=gs://tensorflow2/$4
fi
