#include <cstdio>
#include <ctime>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_CITIES 100
#define MIN_DISTANCE 10
#define MAX_DISTANCE 500
#define NUM_ANTS 100
// #define NUM_ITERATIONS 100 * NUM_CITIES
#define NUM_ITERATIONS 1
#define ALPHA 1
#define BETA 1
#define EVAPORATION_RATE 0.3
#define Q 1
#define MAX_NO_IMPROVEMENT 500

#define cudaCheckError(stmt)                                                   \
  {                                                                            \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA failure %s:%d: '%s'\n", __FILE__, __LINE__,                 \
             cudaGetErrorString(err));                                         \
      exit(1);                                                                 \
    }                                                                          \
  }

void generateDistanceMatrix(int matrix[NUM_CITIES][NUM_CITIES]) {
  for (int i = 0; i < NUM_CITIES; i++) {
    for (int j = 0; j < NUM_CITIES; j++) {
      if (i == j) {
        matrix[i][j] = 0;
      } else if (j > i) {
        matrix[i][j] =
            MIN_DISTANCE + rand() % (MAX_DISTANCE - MIN_DISTANCE + 1);
        matrix[j][i] = matrix[i][j];
      }
    }
  }
}

void printDistanceMatrix(int matrix[NUM_CITIES][NUM_CITIES]) {
  printf("Distance matrix (in km):\n\n");
  printf("    ");
  for (int i = 0; i < NUM_CITIES; i++) {
    printf("C%-4d", i);
  }
  printf("\n");

  for (int i = 0; i < NUM_CITIES; i++) {
    printf("C%-3d", i);
    for (int j = 0; j < NUM_CITIES; j++) {
      printf("%-4d ", matrix[i][j]);
    }
    printf("\n");
  }
}

void printPheromoneMatrix(double matrix[NUM_CITIES][NUM_CITIES]) {
  printf("Pheromone matrix:\n\n");
  printf("    ");
  for (int i = 0; i < NUM_CITIES; i++) {
    printf("C%-8d", i);
  }
  printf("\n");

  for (int i = 0; i < NUM_CITIES; i++) {
    printf("C%-3d", i);
    for (int j = 0; j < NUM_CITIES; j++) {
      printf("%-4f ", matrix[i][j]);
    }
    printf("\n");
  }
}

__global__ void initPheromonesKernel(float *pheromones_d) {
  int i = blockIdx.x;
  int j = threadIdx.x;

  if (j < NUM_CITIES) {
    pheromones_d[i * NUM_CITIES + j] = 1.0;
  }
}

__device__ void decideNext(float *probabilities, curandState state,
                           int *output) {
  float decision = curand_uniform(&state);

  float sum = 0;
  for (int i = 0; i < NUM_CITIES; i++) {
    sum += probabilities[i];
    if (sum >= decision) {
      *output = i;
      return;
    }
  }

  *output = NUM_CITIES - 1;
}

void printPath(int path[NUM_CITIES]) {
  for (int i = 0; i < NUM_CITIES; i++) {
    printf("%d", path[i]);
    if (i != NUM_CITIES - 1) {
      printf(" -> ");
    }
  }
}

__global__ void evaporatePheromonesKernel(float *pheromones_d) {
  int i = blockIdx.x;
  int j = threadIdx.y;

  if (j < NUM_CITIES) {
    pheromones_d[i * NUM_CITIES + j] *= (1 - EVAPORATION_RATE);
  }
}

__global__ void addPheromonesKernel(float *pheromones_d, int *antPaths_d,
                                    int *antPathLengths_d) {
  int ant = blockIdx.x;
  int move = threadIdx.x;

  __shared__ double pathLength;
  pathLength = (float)antPathLengths_d[ant];

  if (move < NUM_CITIES - 1) {
    int source = antPaths_d[ant * NUM_CITIES + move];
    int dest = antPaths_d[ant * NUM_CITIES + move + 1];
    float updateVal = Q / pathLength;
    atomicAdd(pheromones_d + source * NUM_CITIES + dest, updateVal);
    atomicAdd(pheromones_d + dest * NUM_CITIES + source, updateVal);
  } else if (move == NUM_CITIES - 1) {
    int source = antPaths_d[ant * NUM_CITIES + move];
    int dest = antPaths_d[ant * NUM_CITIES];
    float updateVal = Q / pathLength;
    atomicAdd(pheromones_d + source * NUM_CITIES + dest, updateVal);
    atomicAdd(pheromones_d + dest * NUM_CITIES + source, updateVal);
  }
}

__global__ void initAntStatesKernel(int *antPaths_d, int *visited_d) {
  int ant = blockIdx.x;
  int i = threadIdx.x;

  if (i < NUM_ANTS) {
    antPaths_d[ant * NUM_CITIES + i] = i == 0 ? ant : 0;
    visited_d[ant * NUM_CITIES + i] = i == ant;
  }
}

__device__ void setProbabilities(float *probabilities, int *visited_d,
                                 float *pheromones_d, int previousCity,
                                 int *distances_d, int ant, int thread,
                                 int move) {
  int nextCity = threadIdx.x;

  if (nextCity >= NUM_CITIES)
    return;

  if (!visited_d[ant * NUM_CITIES + nextCity]) {
    probabilities[nextCity] =
        pow(pheromones_d[previousCity * NUM_CITIES + nextCity], ALPHA) *
        pow((1 / (float)distances_d[previousCity * NUM_CITIES + nextCity]),
            BETA);
  } else {
    probabilities[nextCity] = 0.0;
  }

  __syncthreads();

  float sumProbabilities = 0.0;
  for (int i = 0; i < NUM_CITIES; i++) {
    sumProbabilities += probabilities[i];
  }

  __syncthreads();

  if (ant == 0 && (thread == 96 || thread == 35 || thread == 20) && move == 1) {
    printf("SUM: %f\n", sumProbabilities);
  }

  probabilities[nextCity] /= sumProbabilities;
}

__global__ void antKernel(int *antPaths_d, int *visited_d, float *pheromones_d,
                          int *distances_d, int *antPathLengths_d) {
  int ant = blockIdx.x;
  int pathLenght = 0;

  // if (ant == 0 && threadIdx.x == 0) {
  //   for (int i = 0; i < NUM_CITIES; i++) {
  //     for (int j = 0; j < NUM_CITIES; j++) {
  //       printf("%f ", pheromones_d[i * NUM_CITIES + j]);
  //     }
  //   }
  // }
  for (int move = 1; move < NUM_CITIES; move++) {
    int previousCity = antPaths_d[ant * NUM_CITIES + move - 1];
    __shared__ float probabilities[NUM_CITIES];

    setProbabilities(probabilities, visited_d, pheromones_d, previousCity,
                     distances_d, ant, threadIdx.x, move);

    __syncthreads();

    if (ant == 0 && threadIdx.x == 0 && (move == 1 || move == 2 || move == 3)) {
      float sum = 0.0;
      for (int i = 0; i < NUM_CITIES; i++) {
        sum += probabilities[i];
        printf("%f ", probabilities[i]);
      }
      printf("\n");
      printf("SUM: %f\n", sum);
    }

    if (threadIdx.x == 0) {
      curandState d_state;
      curand_init(1237, blockIdx.x, 0, &d_state);
      int nextCity;
      decideNext(probabilities, d_state, &nextCity);
      antPaths_d[ant * NUM_CITIES + move] = nextCity;
      visited_d[ant * NUM_CITIES + nextCity] = 1;
      pathLenght += distances_d[previousCity * NUM_CITIES + nextCity];
    }
  }

  if (threadIdx.x == 0) {
    pathLenght +=
        distances_d[antPaths_d[ant * NUM_CITIES + NUM_CITIES - 1] * NUM_CITIES +
                    antPaths_d[ant * NUM_CITIES]];
    antPathLengths_d[ant] = pathLenght;
  }
}

void aco(int distances[NUM_CITIES][NUM_CITIES]) {
  int antPaths[NUM_ANTS][NUM_CITIES];
  int antPathLengths[NUM_ANTS];

  int shortestPathLength = INT_MAX;
  int shortestPath[NUM_CITIES];
  int noImprovement = 0;

  int *antPaths_d, *visited_d, *antPathLengths_d;
  int sizePerAnt = NUM_ANTS * NUM_CITIES * sizeof(int);

  float *pheromones_d;
  int *distances_d;
  int sizePerCityInt = NUM_CITIES * NUM_CITIES * sizeof(int);
  int sizePerCityFloat = NUM_CITIES * NUM_CITIES * sizeof(float);

  // ALLOCATE MEMORY FOR CUDA
  cudaCheckError(cudaMalloc((void **)&antPaths_d, sizePerAnt));
  cudaCheckError(cudaMalloc((void **)&visited_d, sizePerAnt));
  cudaCheckError(
      cudaMalloc((void **)&antPathLengths_d, NUM_ANTS * sizeof(int)));

  cudaCheckError(cudaMalloc((void **)&pheromones_d, sizePerCityFloat));
  cudaCheckError(cudaMalloc((void **)&distances_d, sizePerCityInt));

  // COPY
  cudaCheckError(cudaMemcpy(distances_d, distances, sizePerCityInt,
                            cudaMemcpyHostToDevice));

  initPheromonesKernel<<<NUM_CITIES, 128>>>(pheromones_d);

  for (int iterNum = 0; iterNum < NUM_ITERATIONS; iterNum++) {

    initAntStatesKernel<<<NUM_ANTS, 128>>>(antPaths_d, visited_d);

    antKernel<<<NUM_ANTS, 128>>>(antPaths_d, visited_d, pheromones_d,
                                 distances_d, antPathLengths_d);

    cudaCheckError(
        cudaMemcpy(antPaths, antPaths_d, sizePerAnt, cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(antPathLengths, antPathLengths_d,
                              NUM_ANTS * sizeof(int), cudaMemcpyDeviceToHost));

    int bestAnt = -1;
    for (int ant = 0; ant < NUM_ANTS; ant++) {
      int pathLength = antPathLengths[ant];
      if (pathLength < shortestPathLength) {
        shortestPathLength = pathLength;
        bestAnt = ant;
        noImprovement = 0;
      }
    }
    if (bestAnt != -1) {
      for (int i = 0; i < NUM_CITIES; i++) {
        shortestPath[i] = antPaths[bestAnt][i];
      }
    }

    if (noImprovement > MAX_NO_IMPROVEMENT) {
      printf("Convergence on iter %d\n", iterNum);
      break;
    }

    noImprovement++;

    evaporatePheromonesKernel<<<NUM_CITIES, 128>>>(pheromones_d);
    addPheromonesKernel<<<NUM_ANTS, 128>>>(pheromones_d, antPaths_d,
                                           antPathLengths_d);
  }

  // FREE MEMORY ON DEVICE
  cudaFree(antPaths_d);
  cudaFree(visited_d);
  cudaFree(antPathLengths_d);

  cudaFree(pheromones_d);
  cudaFree(distances_d);

  printf("Shortest: %d\n", shortestPathLength);
  printPath(shortestPath);
}

int main() {
  int distances[NUM_CITIES][NUM_CITIES];
  generateDistanceMatrix(distances);
  clock_t start = clock();
  aco(distances);
  clock_t end = clock();
  printf("\nCuda time: %f\n", (float)(end - start) / CLOCKS_PER_SEC);

  return 0;
}
