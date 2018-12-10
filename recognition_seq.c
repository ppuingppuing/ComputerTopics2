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
  biases[0] = weights[0] + size * IMG_SIZE;
  // 2. Hidden layers
  for(i = 1; i < depth; i++)
  {
    weights[i] = network + (size * IMG_SIZE + size) + (size * size + size) * (i-1);
    biases[i] = weights[i] + size * size;
  }
  // 3. Output layer
  weights[depth] = weights[depth - 1] + size * size + size;
  biases[depth] = weights[depth] + DIGIT_COUNT * size;

  // variables in for
  float * input;
  float output[DIGIT_COUNT];
  float sum = 0;

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
    for(x = 0; x < size; ++x)
    {
      sum = 0; //we should reset sum here.
      cmVar1 = IMG_SIZE * x;
      for(y = 0; y < IMG_SIZE-1; y+=8)
      {
        sum += input[y] * weights[0][cmVar1 + y];
        sum += input[y+1] * weights[0][cmVar1 + y + 1];
        sum += input[y+2] * weights[0][cmVar1 + y + 2];
        sum += input[y+3] * weights[0][cmVar1 + y + 3];
        sum += input[y+4] * weights[0][cmVar1 + y + 4];
        sum += input[y+5] * weights[0][cmVar1 + y + 5];
        sum += input[y+6] * weights[0][cmVar1 + y + 6];
        sum += input[y+7] * weights[0][cmVar1 + y + 7];
      }
      for(;y<IMG_SIZE;++y)
          sum += input[y] * weights[0][cmVar1+y];

      sum += biases[0][x];
      hidden_layers[x] = sigmoid(sum);
    }
    clock_gettime(CLOCK_MONOTONIC,&forE);
    for1_s += (forE.tv_sec - forS.tv_sec) + 1e-9 * (forE.tv_nsec - forS.tv_nsec);

    // Between hidden layers
    clock_gettime(CLOCK_MONOTONIC,&forS);
    for(j = 1; j < depth; ++j)
    {

      cmVar1 = size == 64 ? (j-1) << 6 : size * (j-1);
      for(x = 0; x < size; ++x)
      {
       sum = 0; //we should reset sum here.
       cmVar2 = size == 64 ? x << 6 : size * x;
        for(y = 0; y < size-1; y+=8)
        {
          sum += hidden_layers[cmVar1 + y] * weights[j][cmVar2 + y];
          sum += hidden_layers[cmVar1 + y + 1] * weights[j][cmVar2 + y + 1];
          sum += hidden_layers[cmVar1 + y + 2] * weights[j][cmVar2 + y + 2];
          sum += hidden_layers[cmVar1 + y + 3] * weights[j][cmVar2 + y + 3];
          sum += hidden_layers[cmVar1 + y + 4] * weights[j][cmVar2 + y + 4];
          sum += hidden_layers[cmVar1 + y + 5] * weights[j][cmVar2 + y + 5];
          sum += hidden_layers[cmVar1 + y + 6] * weights[j][cmVar2 + y + 6];
          sum += hidden_layers[cmVar1 + y + 7] * weights[j][cmVar2 + y + 7];
        }
        for(;y<size;++y)
            sum += hidden_layers[cmVar1+y] * weights[j][cmVar2 + y];

        sum += biases[j][x];
        hidden_layers[cmVar1 + size + x] = sigmoid(sum);
      }
    }
    clock_gettime(CLOCK_MONOTONIC,&forE);
    for2_s += (forE.tv_sec - forS.tv_sec) + 1e-9 * (forE.tv_nsec - forS.tv_nsec);
    
    // From the last hidden layer to the output layer 
    clock_gettime(CLOCK_MONOTONIC,&forS);
    for(x = 0; x < DIGIT_COUNT; ++x)
    {
      sum = 0; //we should reset sum here.
      cmVar1 = size==64 ? x << 6 : size * x;
      for(y = 0; y < size-1; y+=8)
      {
        sum += hidden_layers[sizedepth - size + y] * weights[depth][cmVar1 + y];
        sum += hidden_layers[sizedepth - size + y + 1] * weights[depth][cmVar1 + y + 1];
        sum += hidden_layers[sizedepth - size + y + 2] * weights[depth][cmVar1 + y + 2];
        sum += hidden_layers[sizedepth - size + y + 3] * weights[depth][cmVar1 + y + 3];
        sum += hidden_layers[sizedepth - size + y + 4] * weights[depth][cmVar1 + y + 4];
        sum += hidden_layers[sizedepth - size + y + 5] * weights[depth][cmVar1 + y + 5];
        sum += hidden_layers[sizedepth - size + y + 6] * weights[depth][cmVar1 + y + 6];
        sum += hidden_layers[sizedepth - size + y + 7] * weights[depth][cmVar1 + y + 7];
      }
      for(;y<size;++y)
          sum += hidden_layers[sizedepth - size + y]*weights[depth][cmVar1+y];

      sum += biases[depth][x];
      output[x] = sigmoid(sum);
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
