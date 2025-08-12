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
  // int distances[NUM_CITIES][NUM_CITIES] = {
  //     {0,   1050, 1050, 1200, 1350, 1400, 1100, 1050, 600,  1000,
  //      800, 500,  300,  340,  1050, 850,  450,  500,  1400, 1100},
  //     {1050, 0,   350, 530,  650,  880,  1160, 1050, 850,  580,
  //      550,  650, 720, 1000, 2000, 1600, 1100, 950,  1470, 1250},
  //     {1050, 350, 0,   250,  530,  550,  700,  820, 640,  350,
  //      530,  820, 770, 1100, 2000, 1500, 1150, 800, 1200, 1000},
  //     {1200, 530,  250,  0,    243,  380,  560,  710, 750,  430,
  //      630,  1050, 1000, 1350, 2000, 1400, 1000, 900, 1100, 920},
  //     {1350, 650,  530,  243,  0,    350,  700,  750,  850,  550,
  //      800,  1300, 1250, 1600, 2000, 1600, 1300, 1200, 1300, 1200},
  //     {1400, 880,  550,  380,  350,  0,    375,  520,  750, 580,
  //      800,  1300, 1250, 1600, 2000, 1600, 1200, 1050, 900, 850},
  //     {1100, 1160, 700,  560,  700,  375,  0,   275, 530, 570,
  //      830,  1300, 1250, 1600, 1700, 1300, 660, 540, 530, 260},
  //     {1050, 1050, 820,  710,  750,  520, 275, 0,   217, 550,
  //      770,  1100, 1000, 1400, 1400, 850, 400, 300, 570, 300},
  //     {600, 850, 640, 750, 850,  750,  530, 217, 0,   315,
  //      410, 720, 660, 990, 1400, 1200, 400, 280, 710, 540},
  //     {1000, 580, 350, 430,  550,  580,  570, 550, 315, 0,
  //      393,  750, 690, 1000, 1500, 1400, 680, 480, 900, 700},
  //     {800, 550, 530, 630, 800,  800,  830, 770, 410,  393,
  //      0,   450, 380, 730, 1700, 1300, 600, 500, 1100, 900},
  //     {500, 650, 820, 1050, 1300, 1300, 1300, 1100, 720,  750,
  //      450, 0,   170, 360,  1600, 1400, 760,  850,  1600, 1400},
  //     {300, 720, 770, 1000, 1250, 1250, 1250, 1000, 660,  690,
  //      380, 170, 0,   320,  1500, 1300, 640,  610,  1350, 1150},
  //     {340, 1000, 1100, 1350, 1600, 1600, 1600, 1400, 990,  1000,
  //      730, 360,  320,  0,    1400, 1500, 890,  1000, 1500, 1300},
  //     {1050, 2000, 2000, 2000, 2000, 2000, 1700, 1400, 1400, 1500,
  //      1700, 1600, 1500, 1400, 0,    620,  1100, 1300, 1500, 1400},
  //     {850,  1600, 1500, 1400, 1600, 1600, 1300, 850, 1200, 1400,
  //      1300, 1400, 1300, 1500, 620,  0,    660,  850, 1100, 1200},
  //     {450, 1100, 1150, 1000, 1300, 1200, 660, 400, 400, 680,
  //      600, 760,  640,  890,  1100, 660,  0,   140, 760, 760},
  //     {500, 950, 800, 900,  1200, 1050, 540, 300, 280, 480,
  //      500, 850, 610, 1000, 1300, 850,  140, 0,   720, 670},
  //     {1400, 1470, 1200, 1100, 1300, 900,  530, 570, 710, 900,
  //      1100, 1600, 1350, 1500, 1500, 1100, 760, 720, 0,   280},
  //     {1100, 1250, 1000, 920,  1200, 850,  260, 300, 540, 700,
  //      900,  1400, 1150, 1300, 1400, 1200, 760, 670, 280, 0}};
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
