package main

/*
#define _GNU_SOURCE
#include <sched.h>
#include <pthread.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/syscall.h>

static int result_core8 = -1;

void* worker(void* arg) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(8, &set);
    pid_t tid = syscall(SYS_gettid);
    int ret = sched_setaffinity(tid, sizeof(set), &set);
    result_core8 = ret;
    printf("C pthread: sched_setaffinity(tid=%d, core 8) = %d\n", tid, ret);
    return NULL;
}

int try_pthread_core8() {
    pthread_t t;
    pthread_create(&t, NULL, worker, NULL);
    pthread_join(t, NULL);
    return result_core8;
}
*/
import "C"
import "fmt"

func main() {
	fmt.Println("Testing sched_setaffinity from a C pthread:")
	ret := C.try_pthread_core8()
	fmt.Printf("Result: %d (0=success)\n", ret)
}
