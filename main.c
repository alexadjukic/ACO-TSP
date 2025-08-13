#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_CITIES 100
#define MIN_DISTANCE 10
#define MAX_DISTANCE 500
#define NUM_ANTS 100
#define NUM_ITERATIONS 100 * NUM_CITIES
#define ALPHA 1
#define BETA 1
#define EVAPORATION_RATE 0.3
#define Q 1
#define MAX_NO_IMPROVEMENT 500

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

void initPheromones(double pheromones[NUM_CITIES][NUM_CITIES]) {
  for (int i = 0; i < NUM_CITIES; i++) {
    for (int j = 0; j < NUM_CITIES; j++) {
      pheromones[i][j] = 1;
    }
  }
}

void initAntStates(int antPaths[NUM_ANTS][NUM_CITIES],
                   int visited[NUM_ANTS][NUM_CITIES]) {
  for (int i = 0; i < NUM_ANTS; i++) {
    antPaths[i][0] = i;
    for (int j = 0; j < NUM_CITIES; j++) {
      if (i == j) {
        visited[i][j] = 1;
      } else {
        visited[i][j] = 0;
      }
    }
  }
}

double sumArray(double *arr, int size) {
  double result = 0;
  for (int i = 0; i < size; i++) {
    result += arr[i];
  }
  return result;
}

void debugProbabilities(double probabilities[NUM_CITIES]) {
  for (int i = 0; i < NUM_CITIES; i++) {
    printf("Prob %d: %lf\n", i, probabilities[i]);
  }
}

void debugVisited(int visited[NUM_ANTS][NUM_CITIES]) {
  for (int i = 0; i < NUM_ANTS; i++) {
    for (int j = 0; j < NUM_CITIES; j++) {
      printf("Visited[%d][%d] = %d\n", i, j, visited[i][j]);
    }
  }
}

void setProbabilities(double probabilities[NUM_CITIES], int visited[NUM_CITIES],
                      double pheromones[NUM_CITIES][NUM_CITIES],
                      int previousCity, int distances[NUM_CITIES][NUM_CITIES]) {
  for (int nextCity = 0; nextCity < NUM_CITIES; nextCity++) {
    if (!visited[nextCity]) {
      probabilities[nextCity] =
          pow(pheromones[previousCity][nextCity], ALPHA) *
          pow((1 / (double)distances[previousCity][nextCity]), BETA);
    }
  }

  double sumProbabilities = sumArray(probabilities, NUM_CITIES);

  for (int i = 0; i < NUM_CITIES; i++) {
    probabilities[i] /= sumProbabilities;
  }
}

int decideNext(double probabilities[NUM_CITIES]) {
  double decision = rand() / (double)RAND_MAX;

  double sum = 0;
  for (int i = 0; i < NUM_CITIES; i++) {
    sum += probabilities[i];
    if (sum >= decision) {
      return i;
    }
  }

  return NUM_CITIES - 1;
}

void printPath(int path[NUM_CITIES]) {
  for (int i = 0; i < NUM_CITIES; i++) {
    printf("%d", path[i]);
    if (i != NUM_CITIES - 1) {
      printf(" -> ");
    }
  }
}

void evaporatePheromones(double pheromones[NUM_CITIES][NUM_CITIES]) {
  for (int i = 0; i < NUM_CITIES; i++) {
    for (int j = 0; j < NUM_CITIES; j++) {
      pheromones[i][j] *= (1 - EVAPORATION_RATE);
    }
  }
}

void addPheromones(double pheromones[NUM_CITIES][NUM_CITIES],
                   int antPaths[NUM_ANTS][NUM_CITIES],
                   int antPathLengths[NUM_ANTS]) {
  for (int ant = 0; ant < NUM_ANTS; ant++) {
    for (int move = 0; move < NUM_CITIES - 1; move++) {
      int source = antPaths[ant][move];
      int dest = antPaths[ant][move + 1];
      pheromones[source][dest] += Q / (double)antPathLengths[ant];
      pheromones[dest][source] = pheromones[source][dest];
    }
    int source = antPaths[ant][NUM_CITIES - 1];
    int dest = antPaths[ant][0];
    pheromones[source][dest] += Q / (double)antPathLengths[ant];
    pheromones[dest][source] = pheromones[source][dest];
  }
}

void aco(int distances[NUM_CITIES][NUM_CITIES]) {
  double pheromones[NUM_CITIES][NUM_CITIES];
  initPheromones(pheromones);

  int antPaths[NUM_ANTS][NUM_CITIES];
  int antPathLengths[NUM_ANTS];
  int visited[NUM_ANTS][NUM_CITIES];

  int shortestPathLength = INT_MAX;
  int shortestPath[NUM_CITIES];
  int noImprovement = 0;

  for (int iterNum = 0; iterNum < NUM_ITERATIONS; iterNum++) {
    initAntStates(antPaths, visited);
    for (int ant = 0; ant < NUM_ANTS; ant++) {
      int pathLenght = 0;
      for (int move = 1; move < NUM_CITIES; move++) {
        int previousCity = antPaths[ant][move - 1];
        double probabilities[NUM_CITIES] = {0.0};

        setProbabilities(probabilities, visited[ant], pheromones, previousCity,
                         distances);

        int nextCity = decideNext(probabilities);
        antPaths[ant][move] = nextCity;
        visited[ant][nextCity] = 1;
        pathLenght += distances[previousCity][nextCity];
      }

      pathLenght += distances[antPaths[ant][NUM_CITIES - 1]][antPaths[ant][0]];
      antPathLengths[ant] = pathLenght;

      if (pathLenght < shortestPathLength) {
        noImprovement = 0;
        shortestPathLength = pathLenght;
        for (int i = 0; i < NUM_CITIES; i++) {
          shortestPath[i] = antPaths[ant][i];
        }
      }
    }

    if (noImprovement > MAX_NO_IMPROVEMENT) {
      printf("Convergence on iter %d\n", iterNum);
      break;
    }

    noImprovement++;

    evaporatePheromones(pheromones);
    addPheromones(pheromones, antPaths, antPathLengths);
  }

  printf("Shortest: %d\n", shortestPathLength);
  printPath(shortestPath);
}

void setPheromoneUpdate(double pheromoneUpdate[NUM_CITIES][NUM_CITIES],
                        int path[NUM_CITIES], double pathLenght) {
  for (int move = 0; move < NUM_CITIES - 1; move++) {
    int source = path[move];
    int dest = path[move + 1];
    pheromoneUpdate[source][dest] += Q / (double)pathLenght;
    pheromoneUpdate[dest][source] = pheromoneUpdate[source][dest];
  }

  int source = path[NUM_CITIES - 1];
  int dest = path[0];
  pheromoneUpdate[source][dest] += Q / (double)pathLenght;
  pheromoneUpdate[dest][source] = pheromoneUpdate[source][dest];
}

void addPheromoneUpdate(double pheromones[NUM_CITIES][NUM_CITIES],
                        double pheromoneUpdate[NUM_CITIES][NUM_CITIES]) {
  for (int i = 0; i < NUM_CITIES; i++) {
    for (int j = 0; j < NUM_CITIES; j++) {
      pheromones[i][j] += pheromoneUpdate[i][j];
    }
  }
}

void acoParallel(int distances[NUM_CITIES][NUM_CITIES]) {
  double pheromones[NUM_CITIES][NUM_CITIES];
  initPheromones(pheromones);

  int shortestPathLength = INT_MAX;
  int shortestPath[NUM_CITIES];
  int noImprovement = 0;

  for (int iterNum = 0; iterNum < NUM_ITERATIONS; iterNum++) {
    double pheromoneUpdate[NUM_CITIES][NUM_CITIES] = {0.0};

#pragma omp parallel for reduction(+ : pheromoneUpdate)
    for (int ant = 0; ant < NUM_ANTS; ant++) {
      int path[NUM_CITIES];
      int visited[NUM_CITIES] = {0};
      int pathLength = 0;

      path[0] = ant;
      visited[ant] = 1;

      for (int move = 1; move < NUM_CITIES; move++) {
        int previousCity = path[move - 1];
        double probabilities[NUM_CITIES] = {0.0};

        setProbabilities(probabilities, visited, pheromones, previousCity,
                         distances);

        int nextCity = decideNext(probabilities);
        path[move] = nextCity;
        visited[nextCity] = 1;
        pathLength += distances[previousCity][nextCity];
      }

      pathLength += distances[path[NUM_CITIES - 1]][path[0]];

      setPheromoneUpdate(pheromones, path, pathLength);

#pragma omp critical
      {
        if (pathLength < shortestPathLength) {
          noImprovement = 0;
          shortestPathLength = pathLength;
          for (int i = 0; i < NUM_CITIES; i++) {
            shortestPath[i] = path[i];
          }
        }
      }
    }

    if (noImprovement > MAX_NO_IMPROVEMENT)
      break;

    noImprovement++;

    evaporatePheromones(pheromones);
    addPheromoneUpdate(pheromones, pheromoneUpdate);
  }

  printf("Shortest: %d\n", shortestPathLength);
  printPath(shortestPath);
}

int main() {
  int distances[NUM_CITIES][NUM_CITIES];
  generateDistanceMatrix(distances);
  double start = omp_get_wtime();
  aco(distances);
  double end = omp_get_wtime();
  printf("\nSequential time: %lf\n", end - start);
  start = omp_get_wtime();
  acoParallel(distances);
  end = omp_get_wtime();
  printf("\nParallel time: %lf\n", end - start);

  return 0;
}
