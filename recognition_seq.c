#include <stdio.h>
#include <stdlib.h>
#include "recognition.h"
#include <math.h>

#define sigmoid(x) (1 / (1 + exp(-x)))

void recognition(float * images, float * network, int depth, int size, int * labels, float * confidences)
{
  clock_gettime(CLOCK_MONOTONIC, &recS);

  int i, j, x, y;
  float *hidden_layers, **weights, **biases;

  int sizedepth = depth<<6;
  int sizeIMG_SIZE = IMG_SIZE<<6;
  int sizesize = 1<<12;
  int DIGIT_COUNTsize = DIGIT_COUNT << 6;

  hidden_layers = (float *)malloc(sizeof(float) * size * depth);
  weights = (float **)malloc(sizeof(float *) * (depth + 1));
  biases = (float **)malloc(sizeof(float *) * (depth + 1));

  // Set pointers for weights and biases
  // 1. Input layer
  weights[0] = network;
  biases[0] = weights[0] + sizeIMG_SIZE;
  // 2. Hidden layers
  for(i = 1; i < depth; i++)
  {
    weights[i] = network + (sizeIMG_SIZE + size) + (sizesize + size) * (i-1);
    biases[i] = weights[i] + sizesize;
  }
  // 3. Output layer
  weights[depth] = weights[depth - 1] + sizesize + size;
  biases[depth] = weights[depth] + DIGIT_COUNTsize;

  // variables in for
  float * input;
  float output[DIGIT_COUNT];
  //float sum = 0;
  float32x4_t sum;
  float32x4_t Avec,Bvec;

  int cmVar1 = 0; //variable for code motion
  int cmVar2 = 0; //variable for code motion

  float max = 0;
  int label = 0;
  
  // Recognize numbers
  for(i = 0; i < IMG_COUNT; ++i)
  {
    input = images + IMG_SIZE * i;

    // From the input layer to the first hidden layer
    clock_gettime(CLOCK_MONOTONIC,&forS);
    #pragma omp parallel for private(y, cmVar1, Avec, Bvec, sum)
    for(x = 0; x < size; ++x)
    {
      //sum = 0; //we should reset sum here.
      sum = vdupq_n_f32(0);
      cmVar1 = IMG_SIZE * x;
      for(y = 0; y < IMG_SIZE-1; y+=4)
      {
	Avec = vld1q_f32(&input[y]);
	Bvec = vld1q_f32(&weights[0][cmVar1+y]);

	sum = vmlaq_f32(sum,Avec,Bvec);
      }
      for(;y<IMG_SIZE;++y)
          sum[0] += input[y] * weights[0][cmVar1+y];
      sum[0] += sum[1]+sum[2]+sum[3];
      sum[0] += biases[0][x];
      hidden_layers[x] = sigmoid(sum[0]);
    }
    clock_gettime(CLOCK_MONOTONIC,&forE);
    for1_s += (forE.tv_sec - forS.tv_sec) + 1e-9 * (forE.tv_nsec - forS.tv_nsec);

    // Between hidden layers
    clock_gettime(CLOCK_MONOTONIC,&forS);
    for(j = 1; j < depth; ++j)
    {

      cmVar1 = size == 64 ? (j-1) << 6 : size * (j-1);

     #pragma omp parallel for private(y, cmVar2, Avec, Bvec, sum)
      for(x = 0; x < size; ++x)
      {
       //sum = 0; //we should reset sum here.
       sum = vdupq_n_f32(0);
       cmVar2 = size == 64 ? x << 6 : size * x;
        for(y = 0; y < size-1; y+=4)
        {
	  Avec = vld1q_f32(&hidden_layers[cmVar1 + y]);
	  Bvec = vld1q_f32(&weights[j][cmVar2+y]);

	  sum = vmlaq_f32(sum, Avec, Bvec);  
        }
        for(;y<size;++y)
            sum[0] += hidden_layers[cmVar1+y] * weights[j][cmVar2 + y];

        sum[0] += sum[1]+sum[2]+sum[3];
        sum[0] += biases[j][x];
        hidden_layers[cmVar1 + size + x] = sigmoid(sum[0]);
      }
    }
    clock_gettime(CLOCK_MONOTONIC,&forE);
    for2_s += (forE.tv_sec - forS.tv_sec) + 1e-9 * (forE.tv_nsec - forS.tv_nsec);
    
    // From the last hidden layer to the output layer 
    clock_gettime(CLOCK_MONOTONIC,&forS);
    for(x = 0; x < DIGIT_COUNT; ++x)
    {
      //sum = 0; //we should reset sum here.
      sum = vdupq_n_f32(0);
      cmVar1 = size==64 ? x << 6 : size * x;
      for(y = 0; y < size-1; y+=4)
      {
        Avec = vld1q_f32(&hidden_layers[sizedepth - size + y]);
        Bvec = vld1q_f32(&weights[depth][cmVar1+y]);

	sum = vmlaq_f32(sum, Avec, Bvec);  
      }
      for(;y<size;++y)
          sum[0] += hidden_layers[sizedepth - size + y]*weights[depth][cmVar1+y];

      sum[0] += sum[1]+sum[2]+sum[3];
      sum[0] += biases[depth][x];
      output[x] = sigmoid(sum[0]);
    }
    clock_gettime(CLOCK_MONOTONIC,&forE);
    for3_s += (forE.tv_sec - forS.tv_sec) + 1e-9 * (forE.tv_nsec - forS.tv_nsec);

    // Find the answer
    clock_gettime(CLOCK_MONOTONIC,&forS);
    max = 0;
    label = 0;
    for(x = 0; x < DIGIT_COUNT; ++x)
    {
      if(output[x] > max)
      {
        label = x;
        max = output[x];
      }
    }    


    // Store the result
    confidences[i] = max;
    labels[i] = label;
    clock_gettime(CLOCK_MONOTONIC,&forE);
    for4_s += (forE.tv_sec - forS.tv_sec) + 1e-9 * (forE.tv_nsec - forS.tv_nsec);
  }
  
  clock_gettime(CLOCK_MONOTONIC, &recE);
  sec2 += (recE.tv_sec - recS.tv_sec) + 1e-9 * (recE.tv_nsec - recS.tv_nsec);
  sec2_count ++;
}
