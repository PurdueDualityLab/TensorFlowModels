# TPU Utilities

This folder contains resources to help with generating TPUs outside of the
Google Cloud interface. It only exists to prevent costly mis-clicks in the UI
by eliminating any TPUs that we are not authorized to use. As such, this folder
is meant for internal use. Use the `gcloud compute` command or the Cloud Console
and Shell instead to manage your TPU and VMs if you are not a part of the team.

```sh
gcloud compute help
gcloud compute instances help
gcloud compute tpus help
```
