
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "recognition.h"

struct timespec mainS, mainE, forS;
struct timespec recS, recE, forE;

double sec1, sec2;
int sec1_count, sec2_count;

double for1_s, for2_s, for3_s, for4_s;

int timespec_subtract(struct timespec*, struct timespec*, struct timespec*);
void load_MNIST(float * images, int * labels);

int main(int argc, char** argv) {

  clock_gettime(CLOCK_MONOTONIC, &mainS);

  float *images, *network, *confidences, accuracy;
  int *labels;
  int *labels_ans;
  int i, correct, total_network_size;
  FILE *io_file;
  struct timespec start, end, spent;

  int size = 4096, depth = 3;

  // Check parameters
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <network file> <output file>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  images = (float *)malloc(sizeof(float)*IMG_COUNT*IMG_SIZE);
  labels = (int *)malloc(sizeof(int)*IMG_COUNT);
  labels_ans = (int *)malloc(sizeof(int)*IMG_COUNT);
  confidences = (float *)malloc(sizeof(float)*IMG_COUNT);

  io_file = fopen(argv[1], "r");
  if(!io_file)
  {
    fprintf(stderr, "Invalid network file %s!\n", argv[1]);
    exit(EXIT_FAILURE);
  }
  fread(&depth, sizeof(int), 1, io_file);
  fread(&size, sizeof(int), 1, io_file);
  printf("size=%d, depth=%d\n", size, depth);
  total_network_size = (IMG_SIZE * size + size) + (depth - 1) * (size * size + size) + size  * DIGIT_COUNT + DIGIT_COUNT;
  network = (float *)malloc(sizeof(float) * (total_network_size));
  fread(network, sizeof(float), total_network_size, io_file);
  fclose(io_file);

  io_file = fopen("MNIST_image.bin", "r");
  fread(images, sizeof(float), IMG_COUNT * IMG_SIZE, io_file); 
  fclose(io_file);

  io_file = fopen("MNIST_label.bin", "r");
  fread(labels_ans, sizeof(int), IMG_COUNT, io_file); 
  fclose(io_file);

  clock_gettime(CLOCK_MONOTONIC, &start);
  recognition(images, network, depth, size, labels, confidences);
  clock_gettime(CLOCK_MONOTONIC, &end);
  timespec_subtract(&spent, &end, &start);

  correct = 0;
  for(i = 0; i <IMG_COUNT; i++)
  {
    if(labels_ans[i] == labels[i]) correct++;
  }
  accuracy = (float)correct / (float)IMG_COUNT;

  printf("Elapsed time: %ld.%03ld sec\n", spent.tv_sec, spent.tv_nsec/1000/1000);
  printf("Accuracy: %.3f\n", accuracy);
  // Write the result
  io_file = fopen(argv[2], "wb");
  fprintf(io_file, "%.3f\n", accuracy);
  for(i = 0; i < IMG_COUNT; i++)
  {
    fprintf(io_file,"%d, %d, %.3f\n", labels_ans[i], labels[i], confidences[i]);
  }
  fclose(io_file);

  clock_gettime(CLOCK_MONOTONIC, &mainE);
  sec1 += (mainE.tv_sec - mainS.tv_sec) + 1e-9 * (mainE.tv_nsec - mainS.tv_nsec);
  sec1_count ++;

  // Ref : HW1
  printf("%-20s %s : %.9lf (%d)\n","main", "[ms/call] (n called)",1000*(sec1-sec2),sec1_count);
        
  printf("%-20s %s : %.9lf (%d)\n","recognition", "[ms/call] (n called)",1000*(sec2),sec2_count);
               

  printf("---------------------------------------------\n");
  printf("%-20s %s : %.9lf (%d)\n","LOOP1_time", "[ms/call] (n called)",1000*(for1_s),IMG_COUNT);
  printf("%-20s %s : %.9lf (%d)\n","LOOP2_time", "[ms/call] (n called)",1000*(for2_s),IMG_COUNT);
  printf("%-20s %s : %.9lf (%d)\n","LOOP3_time", "[ms/call] (n called)",1000*(for3_s),IMG_COUNT);
  printf("%-20s %s : %.9lf (%d)\n","LOOP4_time", "[ms/call] (n called)",1000*(for4_s),IMG_COUNT);


  return 0;
}

int timespec_subtract(struct timespec* result, struct timespec *x, struct timespec *y) {
  /* Perform the carry for the later subtraction by updating y. */
  if (x->tv_nsec < y->tv_nsec) {
    int nsec = (y->tv_nsec - x->tv_nsec) / 1000000000 + 1;
    y->tv_nsec -= 1000000000 * nsec;
    y->tv_sec += nsec;
  }
  if (x->tv_nsec - y->tv_nsec > 1000000000) {
    int nsec = (x->tv_nsec - y->tv_nsec) / 1000000000;
    y->tv_nsec += 1000000000 * nsec;
    y->tv_sec -= nsec;
  }

  /* Compute the time remaining to wait.
     tv_nsec is certainly positive. */
  result->tv_sec = x->tv_sec - y->tv_sec;
  result->tv_nsec = x->tv_nsec - y->tv_nsec;

  /* Return 1 if result is negative. */
  return x->tv_sec < y->tv_sec;
}
