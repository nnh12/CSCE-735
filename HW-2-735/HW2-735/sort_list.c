//
// Sorts a list using multiple threads
//

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <limits.h>

#define MAX_THREADS     65536
#define MAX_LIST_SIZE   100000000

#define DEBUG 0

// Barrier used to synchronize threads
typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t cond;
    int count;
    int generation;
    int num_threads;
} barrier_t;

// Thread variables
pthread_t p_threads[MAX_THREADS];
pthread_attr_t attr;
barrier_t barrier;

// Global variables
int num_threads;		// Number of threads to create - user input 
int list_size;			// List size
int *list;			// List of values
int *work;			// Work array
int *list_orig;			// Original list of values, used for error checking
int ptr[MAX_THREADS+1];		// Starting position of each sublist
int q_levels;			// Number of merge levels

// Print list - for debugging
void print_list(int *list, int list_size) {
    int i;
    for (i = 0; i < list_size; i++) {
        printf("[%d] \t %16d\n", i, list[i]); 
    }
    printf("--------------------------------------------------------------------\n"); 
}

// Comparison routine for qsort (stdlib.h) which is used to 
// a thread's sub-list at the start of the algorithm
int compare_int(const void *a0, const void *b0) {
    int a = *(int *)a0;
    int b = *(int *)b0;
    if (a < b) {
        return -1;
    } else if (a > b) {
        return 1;
    } else {
        return 0;
    }
}

// Return index of first element larger than or equal to v in sorted list
// ... return last if all elements are smaller than v
// ... elements in list[first], list[first+1], ... list[last-1]
//
//   int idx = first; while ((v > list[idx]) && (idx < last)) idx++;
//
int binary_search_lt(int v, int *list, int first, int last) {
   
    // Linear search code
    // int idx = first; while ((v > list[idx]) && (idx < last)) idx++; return idx;

    int left = first; 
    int right = last-1; 

    if (list[left] >= v) return left;
    if (list[right] < v) return right+1;
    int mid = (left+right)/2; 
    while (mid > left) {
        if (list[mid] < v) {
	    left = mid; 
	} else {
	    right = mid;
	}
	mid = (left+right)/2;
    }
    return right;
}
// Return index of first element larger than v in sorted list
// ... return last if all elements are smaller than or equal to v
// ... elements in list[first], list[first+1], ... list[last-1]
//
//   int idx = first; while ((v >= list[idx]) && (idx < last)) idx++;
//
int binary_search_le(int v, int *list, int first, int last) {

    // Linear search code
    // int idx = first; while ((v >= list[idx]) && (idx < last)) idx++; return idx;
 
    int left = first; 
    int right = last-1; 

    if (list[left] > v) return left; 
    if (list[right] <= v) return right+1;
    int mid = (left+right)/2; 
    while (mid > left) {
        if (list[mid] <= v) {
	    left = mid; 
	} else {
	    right = mid;
	}
	mid = (left+right)/2;
    }
    return right;
}

// Initialize barrier
void barrier_init(barrier_t *b, int n) {
    b->count = 0;
    b->generation = 0;
    b->num_threads = n;
    pthread_mutex_init(&b->lock, NULL);
    pthread_cond_init(&b->cond, NULL);
}

// Wait for all threads to arrive at barrier
void barrier_wait(barrier_t *b) {
    int my_gen;
    pthread_mutex_lock(&b->lock);
    my_gen = b->generation;
    b->count++;
    if (b->count == b->num_threads) {
        b->count = 0;
        b->generation++;
        pthread_cond_broadcast(&b->cond);
    } else {
        while (my_gen == b->generation) {
            pthread_cond_wait(&b->cond, &b->lock);
        }
    }
    pthread_mutex_unlock(&b->lock);
}

// Thread routine: sorts its own sublist, then merges it into larger
// sublists for each merge level
void *merge_sort_thread(void *arg) {
    int my_id = (int)(long)arg;
    int i, level;
    int my_list_size;
    int my_blk_size, my_own_blk, my_own_idx;
    int my_search_blk, my_search_idx, my_search_idx_max;
    int my_write_blk, my_write_idx;
    int my_search_count;
    int idx, i_write;
    int np = list_size/num_threads; 	// Sub list size 

    // Sort local list
    my_list_size = ptr[my_id+1]-ptr[my_id];
    qsort(&list[ptr[my_id]], my_list_size, sizeof(int), compare_int);

    // Wait until every thread has sorted its sublist
    barrier_wait(&barrier);

    // Sort list
    for (level = 0; level < q_levels; level++) {

        // Each thread scatters its sub_list into work array
        my_blk_size = np * (1 << level); 

        my_own_blk = ((my_id >> level) << level);
        my_own_idx = ptr[my_own_blk];

        my_search_blk = ((my_id >> level) << level) ^ (1 << level);
        my_search_idx = ptr[my_search_blk];
        my_search_idx_max = my_search_idx+my_blk_size;

        my_write_blk = ((my_id >> (level+1)) << (level+1));
        my_write_idx = ptr[my_write_blk];

        idx = my_search_idx;
        my_search_count = 0;

        // Binary search for 1st element
        if (my_search_blk > my_own_blk) {
            idx = binary_search_lt(list[ptr[my_id]], list, my_search_idx, my_search_idx_max); 
        } else {
            idx = binary_search_le(list[ptr[my_id]], list, my_search_idx, my_search_idx_max); 
        }
        my_search_count = idx - my_search_idx;
        i_write = my_write_idx + my_search_count + (ptr[my_id]-my_own_idx); 
        work[i_write] = list[ptr[my_id]];

        // Linear search for 2nd element onwards
        for (i = ptr[my_id]+1; i < ptr[my_id+1]; i++) {
            if (my_search_blk > my_own_blk) {
                while ((list[i] > list[idx]) && (idx < my_search_idx_max)) {
                    idx++; my_search_count++;
                }
            } else {
                while ((list[i] >= list[idx]) && (idx < my_search_idx_max)) {
                    idx++; my_search_count++;
                }
            }
            i_write = my_write_idx + my_search_count + (i-my_own_idx); 
            work[i_write] = list[i];
        }

        // Wait until every thread has written its sublist into work array
        barrier_wait(&barrier);

        // Copy work into list for next iteration
        for (i = ptr[my_id]; i < ptr[my_id+1]; i++) {
            list[i] = work[i];
        } 

        // Wait until every thread has copied its portion before next level
        barrier_wait(&barrier);
    }

    pthread_exit(NULL);
}

// Sort list via parallel merge sort
void sort_list(int q) {

    int my_id;

    q_levels = q;

    // Initialize starting position for each sublist
    for (my_id = 0; my_id < num_threads; my_id++) {
        ptr[my_id] = my_id * (list_size/num_threads);
    }
    ptr[num_threads] = list_size;

    // Create threads
    for (my_id = 0; my_id < num_threads; my_id++) {
        pthread_create(&p_threads[my_id], &attr, merge_sort_thread, (void *)(long)my_id);
    }

    // Wait for all threads to finish
    for (my_id = 0; my_id < num_threads; my_id++) {
        pthread_join(p_threads[my_id], NULL);
    }
}

// Main program - set up list of random integers and use threads to sort the list
//
// Input: 
//	k = log_2(list size), therefore list_size = 2^k
//	q = log_2(num_threads), therefore num_threads = 2^q
//
int main(int argc, char *argv[]) {

    struct timespec start, stop, stop_qsort;
    double total_time, time_res, total_time_qsort;
    int k, q, j, error; 

    // Read input, validate
    if (argc != 3) {
	printf("Need two integers as input \n"); 
	printf("Use: <executable_name> <log_2(list_size)> <log_2(num_threads)>\n"); 
	exit(0);
    }
    k = atoi(argv[argc-2]);
    if ((list_size = (1 << k)) > MAX_LIST_SIZE) {
	printf("Maximum list size allowed: %d.\n", MAX_LIST_SIZE);
	exit(0);
    }; 
    q = atoi(argv[argc-1]);
    if ((num_threads = (1 << q)) > MAX_THREADS) {
	printf("Maximum number of threads allowed: %d.\n", MAX_THREADS);
	exit(0);
    }; 
    if (num_threads > list_size) {
	printf("Number of threads (%d) < list_size (%d) not allowed.\n", 
	   num_threads, list_size);
	exit(0);
    }; 

    // Allocate list, list_orig, and work

    list = (int *) malloc(list_size * sizeof(int));
    list_orig = (int *) malloc(list_size * sizeof(int));
    work = (int *) malloc(list_size * sizeof(int));

// Initialize barrier and thread attributes
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    barrier_init(&barrier, num_threads);

    // Initialize list of random integers; list will be sorted by 
    // multi-threaded parallel merge sort
    // Copy list to list_orig; list_orig will be sorted by qsort and used
    // to check correctness of multi-threaded parallel merge sort
    srand48(0); 	// seed the random number generator
    for (j = 0; j < list_size; j++) {
	list[j] = (int) lrand48();
	list_orig[j] = list[j];
    }
    // duplicate first value at last location to test for repeated values
    list[list_size-1] = list[0]; list_orig[list_size-1] = list_orig[0];

    // Create threads; each thread sorts and merges its share of the list
    clock_gettime(CLOCK_REALTIME, &start);

    sort_list(q);

    // Compute time taken
    clock_gettime(CLOCK_REALTIME, &stop);
    total_time = (stop.tv_sec-start.tv_sec)
	+0.000000001*(stop.tv_nsec-start.tv_nsec);

    // Check answer
    qsort(list_orig, list_size, sizeof(int), compare_int);
    clock_gettime(CLOCK_REALTIME, &stop_qsort);
    total_time_qsort = (stop_qsort.tv_sec-stop.tv_sec)
	+0.000000001*(stop_qsort.tv_nsec-stop.tv_nsec);

    error = 0; 
    for (j = 1; j < list_size; j++) {
	if (list[j] != list_orig[j]) error = 1; 
    }

    if (error != 0) {
	printf("Houston, we have a problem!\n"); 
    }

    // Print time taken
    printf("List Size = %d, Threads = %d, error = %d, time (sec) = %8.4f, qsort_time = %8.4f\n", 
	    list_size, num_threads, error, total_time, total_time_qsort);

// Destroy barrier and thread attributes
    pthread_attr_destroy(&attr);
    pthread_mutex_destroy(&barrier.lock);
    pthread_cond_destroy(&barrier.cond);

    free(list); free(work); free(list_orig); 

}

