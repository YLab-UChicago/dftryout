#include <stdio.h>

#define INPUT_SIZE 20
#define FILTER_SIZE 3
#define OUTPUT_SIZE (INPUT_SIZE - FILTER_SIZE + 1)

void conv(float input[INPUT_SIZE][INPUT_SIZE], 
          float weights[FILTER_SIZE][FILTER_SIZE], 
          float biases[OUTPUT_SIZE][OUTPUT_SIZE], 
          float output[OUTPUT_SIZE][OUTPUT_SIZE]) {
    for (int y = 0; y < OUTPUT_SIZE; ++y) {
        for (int x = 0; x < OUTPUT_SIZE; ++x) {
            float value = 0;
            for (int fy = 0; fy < FILTER_SIZE; ++fy) {
                for (int fx = 0; fx < FILTER_SIZE; ++fx) {
                    int input_index_y = y + fy;
                    int input_index_x = x + fx;
                    value += input[input_index_y][input_index_x] * weights[fy][fx];
                }
            }
            output[y][x] = value + biases[y][x];
        }
    }
}

int main() {
    float input[INPUT_SIZE][INPUT_SIZE];
    float weights[FILTER_SIZE][FILTER_SIZE];
    float biases[OUTPUT_SIZE][OUTPUT_SIZE];
    float output[OUTPUT_SIZE][OUTPUT_SIZE];

    // TODO: Initialize input, weights, and biases here...

    conv(input, weights, biases, output);

    // Print output
    for (int y = 0; y < OUTPUT_SIZE; ++y) {
        for (int x = 0; x < OUTPUT_SIZE; ++x) {
            printf("%.2f ", output[y][x]);
        }
        printf("\n");
    }

    return 0;
}
