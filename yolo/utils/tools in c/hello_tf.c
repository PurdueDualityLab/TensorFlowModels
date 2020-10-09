#include <stdio.h>
#include <tensorflow/c/c_api.h>

int main(){
    printf("items tf: %s", TF_Version());
}