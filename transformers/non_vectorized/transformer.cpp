#include <iostream>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <math.h>
#include <time.h>
#include <iomanip> 
#include <random>
#include <fstream>

#define VOCAB_SIZE 5000
#define EMBEDDING_SIZE 1280
#define SEQUENCE_LENGTH 30
#define NUM_HEADS 2
#define QKV_MATRIX_WIDTH (EMBEDDING_SIZE / NUM_HEADS)
#define FINAL_DIMENSIONALITY 128

void run_transformer();
void run_transformer_optimized();

// compile with: "g++ -std=c++11 -lm optimized_transformer.cpp -o output"
// or "make" if using the makefile.
// Run with ./output <number of runs>
// program crashes if EMBEDDING_SIZE < SEQUENCE_LENGTH or EMBEDDING_SIZE > 561


////////////////////////////////////////////////////////////////////////////////
// SHARED FUNCTIONS
// -initialize_embeddings
// -fill_token_IDs
// -get_embeddings
// -softmax
// -fill_QKV_weights
// -fill_output_weights
// -final_transform
// -time_transformer
////////////////////////////////////////////////////////////////////////////////


// Function to initialize the embedding matrix with random values
void initialize_embeddings(float *embeddings) {
  srand(time(0));
  for (int i = 0; i < VOCAB_SIZE * EMBEDDING_SIZE; i++){
    embeddings[i] = ((float)rand() / (RAND_MAX)) - 0.5;
  }
}


// Get random tokenID values between 0 (inclusive) and VOCAB_SIZE (exclusive)
void fill_token_IDs(int *tokenIDs){
    //Create seed using current time
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

    // Build random number generator
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(0, VOCAB_SIZE - 1);

    for (int i = 0; i < VOCAB_SIZE; i++){
        int randomValue = distribution(generator);
        tokenIDs[i] = randomValue;
    }
}


// Function to look up the embeddings for a given sequence of token indices
void get_embeddings(int *tokens, int sequence_length, float *embeddings, float *output) {
  for (int i = 0; i < sequence_length; i++) {
    int token_index = tokens[i];
    for (int j = 0; j < EMBEDDING_SIZE; j++) {
      output[j + (i * EMBEDDING_SIZE)] = embeddings[j + (token_index * EMBEDDING_SIZE)];
    }
  }
}


// Softmax function to normalize values to a probability distribution
void softmax(float *x, int length) {
  float max = 0.0f;
  float sum = 0.0f;
  for (int i = 0; i < length; i++) {
    if (x[i] > max) {
      max = x[i];
    }
  }
  for (int i = 0; i < length; i++) {
    x[i] = exp(x[i] - max);
    sum += x[i];
  }
  for (int i = 0; i < length; i++) {
    x[i] /= sum;
  }
}


// Initialize all Q/K/V weights to random values
void fill_QKV_weights(float *Q_weights, float *K_weights, float *V_weights){
    srand(time(0));
    for (int i = 0; i < SEQUENCE_LENGTH * QKV_MATRIX_WIDTH; i++){
        Q_weights[i] = ((float)rand() / (RAND_MAX)) - 0.5;
        K_weights[i] = ((float)rand() / (RAND_MAX)) - 0.5;
        V_weights[i] = ((float)rand() / (RAND_MAX)) - 0.5;
    }
}


// Initialize all output weights to random values
void fill_output_weights(float *output_weights){
    srand(time(0));
    for (int i = 0; i < EMBEDDING_SIZE * NUM_HEADS * FINAL_DIMENSIONALITY; i++){
        output_weights[i] = ((float)rand() / (RAND_MAX)) - 0.5;
    }
}


// Perform final transform on concatenated attention matrix
void final_transform(float *result, float *concat_attnts, float *output_weights){
    for (int i = 0; i < FINAL_DIMENSIONALITY; i++){
        for (int j = 0; j < SEQUENCE_LENGTH; j++){
            float partial_sum = 0;
            for (int k = 0; k < EMBEDDING_SIZE; k++){
                partial_sum += concat_attnts[k + (j * EMBEDDING_SIZE)] * output_weights[(k * FINAL_DIMENSIONALITY) + i];
            }
            result[i + (j * FINAL_DIMENSIONALITY)] = partial_sum;
        }
    }
}


// Time transformer; t = 0 for unoptimized ; t = 1 for optimized
std::chrono::duration<double, std::milli> time_transformer(int t) {
    if (!t){ 
        auto start = std::chrono::high_resolution_clock::now();
        run_transformer();
        auto stop = std::chrono::high_resolution_clock::now();
        return (stop - start);
    }
    else {
        auto start = std::chrono::high_resolution_clock::now();
        run_transformer_optimized();
        auto stop = std::chrono::high_resolution_clock::now();
        return (stop - start);
    }
}


////////////////////////////////////////////////////////////////////////////////
// OPTIMIZED FUNCTIONS
// -fill_QKV_matrices_IS
// -attention_os
// -run_transformer_optimized
////////////////////////////////////////////////////////////////////////////////


// Multiplies embeddings by Q/K/V weights to produce Q/K/V matrices using an input stationary dataflow
void fill_QKV_matrices_IS(float *imbeddings,float *Q_weights, float *K_weights, float *V_weights, float *Q, float *K, float *V){
    for (int i = 0; i < QKV_MATRIX_WIDTH; i++){
        for (int j = 0; j < SEQUENCE_LENGTH; j++){
            float Q_sum = 0;
            float K_sum = 0;
            float V_sum = 0;
            for (int k = 0; k < EMBEDDING_SIZE; k++){
                float current_in_val = imbeddings[k + EMBEDDING_SIZE * j];
                Q_sum += current_in_val * Q_weights[k * QKV_MATRIX_WIDTH + i];
                K_sum += current_in_val * K_weights[k * QKV_MATRIX_WIDTH + i];
                V_sum += current_in_val * V_weights[k * QKV_MATRIX_WIDTH + i];
            }
            Q[i + j * QKV_MATRIX_WIDTH] = Q_sum;
            K[i + j * QKV_MATRIX_WIDTH] = K_sum;
            V[i + j * QKV_MATRIX_WIDTH] = V_sum;
        }
    }
}


void attention_OS(float *Q, float *K, float *V, float *concat_attnts, int sequence_length, int embed_size, int current_head){
  float qk[SEQUENCE_LENGTH][SEQUENCE_LENGTH];
  // Calculate dot product of Q and K
  for (int i = 0; i < sequence_length; i++) {
    for (int j = 0; j < sequence_length; j++) {
      qk[i][j] = 0.0f;
      for (int d = 0; d < embed_size; d++) {
        qk[i][j] += Q[i * embed_size + d] * K[j * embed_size + d];
      }
      // Scale by the square root of the embedding size
      qk[i][j] /= sqrt(embed_size);
    }
    // Apply softmax to the scaled values
    softmax(qk[i], sequence_length);
  }

  // Multiply the scaled values by V to produce the attention output and put into concatenated attention matrix
    for (int i = 0; i < sequence_length; i++){
        for (int d = 0; d < embed_size; d++){
            for (int j = 0; j < sequence_length; j++){
                int current_index = d + (i * embed_size * NUM_HEADS) + (current_head * embed_size);
                concat_attnts[current_index] += qk[i][j] * V[j * embed_size + d];
            }
        }
    }
}


void run_transformer_optimized(){

    // Initialize the embeddings
    float *embeddings = (float*)calloc(sizeof(float),VOCAB_SIZE * EMBEDDING_SIZE);
    initialize_embeddings(embeddings);

    // Generate tokens IDs
    int *tokenIDs = (int*)calloc(sizeof(int), VOCAB_SIZE);
    fill_token_IDs(tokenIDs);

    // Get the embeddings for the token sequence
    float *input_embeddings = (float*)calloc(sizeof(float), EMBEDDING_SIZE * SEQUENCE_LENGTH);
    get_embeddings(tokenIDs, SEQUENCE_LENGTH, embeddings, input_embeddings);

    // Initialize output matrix for concatenated attention matrices
    float *concat_attnts = (float*)calloc(sizeof(float), EMBEDDING_SIZE * SEQUENCE_LENGTH * NUM_HEADS);

    // Calculate attention score matrix for each head
    for (int i = 0; i < NUM_HEADS; i++){

        // Fill weight matrices 
        float *Q_weights = (float*)calloc(sizeof(float), EMBEDDING_SIZE * QKV_MATRIX_WIDTH);
        float *K_weights = (float*)calloc(sizeof(float), EMBEDDING_SIZE * QKV_MATRIX_WIDTH);
        float *V_weights = (float*)calloc(sizeof(float), EMBEDDING_SIZE * QKV_MATRIX_WIDTH);
        fill_QKV_weights(Q_weights, K_weights, V_weights);

        // Calculate QKV matrices using input stationary dataflow
        float *Q = (float*)calloc(sizeof(float), SEQUENCE_LENGTH * QKV_MATRIX_WIDTH);
        float *K = (float*)calloc(sizeof(float), SEQUENCE_LENGTH * QKV_MATRIX_WIDTH);
        float *V = (float*)calloc(sizeof(float), SEQUENCE_LENGTH * QKV_MATRIX_WIDTH);
        fill_QKV_matrices_IS(input_embeddings, Q_weights, K_weights, V_weights, Q, K, V);

        // Calculate attention matrix for current head and immediately place output in concatenated matrix
        attention_OS(Q, K, V, concat_attnts, SEQUENCE_LENGTH, EMBEDDING_SIZE, i);

        // Free data
        free(Q_weights);
        free(K_weights);
        free(V_weights);
        free(Q);
        free(K);
        free(V);
    }

    // Create output weight matrix
    float *output_weights = (float*)calloc(sizeof(float), EMBEDDING_SIZE * NUM_HEADS * FINAL_DIMENSIONALITY);
    fill_output_weights(output_weights);

    // Initialize result matrix
    float *result = (float*)calloc(sizeof(float), SEQUENCE_LENGTH * FINAL_DIMENSIONALITY);
    final_transform(result, concat_attnts, output_weights); //TODO: Integrate with attention_OS??

    // Free data
    free(embeddings);
    free(tokenIDs);
    free(input_embeddings);
    free(concat_attnts);
    free(output_weights);
    free(result);

    // DONE
}


////////////////////////////////////////////////////////////////////////////////
// UNOPTIMIZED FUNCTIONS
// -fill_QKV_matrices
// -attention
// -run_transformer
////////////////////////////////////////////////////////////////////////////////


// Multiplies embeddings by Q/K/V weights to produce Q/K/V matrices without a dataflow
void fill_QKV_matrices(float *imbeddings,float *Q_weights, float *K_weights, float *V_weights, float *Q, float *K, float *V){

    //Fill Q matrix
    for (int i = 0; i < QKV_MATRIX_WIDTH; i++){
        for (int j = 0; j < SEQUENCE_LENGTH; j++){
            float Q_sum = 0;
            for (int k = 0; k < EMBEDDING_SIZE; k++){
                Q_sum += imbeddings[k + EMBEDDING_SIZE * j] * Q_weights[k * QKV_MATRIX_WIDTH + i];
            }
            Q[i + j * QKV_MATRIX_WIDTH] = Q_sum;
        }
    }

    // Fill K matrix
    for (int i = 0; i < QKV_MATRIX_WIDTH; i++){
        for (int j = 0; j < SEQUENCE_LENGTH; j++){
            float K_sum = 0;
            for (int k = 0; k < EMBEDDING_SIZE; k++){

                K_sum += imbeddings[k + EMBEDDING_SIZE * j] * K_weights[k * QKV_MATRIX_WIDTH + i];
            }
            K[i + j * QKV_MATRIX_WIDTH] = K_sum;
        }
    }

    // Fill V matrix
    for (int i = 0; i < QKV_MATRIX_WIDTH; i++){
        for (int j = 0; j < SEQUENCE_LENGTH; j++){
            float V_sum = 0;
            for (int k = 0; k < EMBEDDING_SIZE; k++){
                V_sum += imbeddings[k + EMBEDDING_SIZE * j] * V_weights[k * QKV_MATRIX_WIDTH + i];
            }
            V[i + j * QKV_MATRIX_WIDTH] = V_sum;
        }
    }
}


// Attention function implementing the scaled dot-product attention mechanism
void attention(float *Q, float *K, float *V, float *output, int sequence_length, int embed_size){
  float qk[SEQUENCE_LENGTH][SEQUENCE_LENGTH];
  // Calculate dot product of Q and K
  for (int i = 0; i < sequence_length; i++) {
    for (int j = 0; j < sequence_length; j++) {
      qk[i][j] = 0.0f;
      for (int d = 0; d < embed_size; d++) {
        qk[i][j] += Q[i * embed_size + d] * K[j * embed_size + d];
      }
      // Scale by the square root of the embedding size
      qk[i][j] /= sqrt(embed_size);
    }
    // Apply softmax to the scaled values
    softmax(qk[i], sequence_length);
  }

  // Multiply the scaled values by V to produce the attention output
  for (int i = 0; i < sequence_length; i++) {
    for (int d = 0; d < embed_size; d++) {
    //   output[i * embed_size + d] = 0.0f;
      for (int j = 0; j < sequence_length; j++) {
        int current_index = i * embed_size + d;
        output[current_index] += qk[i][j] * V[j * embed_size + d];
      }
    }
  }
}


void run_transformer(){

    // Initialize the embeddings
    float *embeddings = (float*)calloc(sizeof(float),VOCAB_SIZE * EMBEDDING_SIZE);
    initialize_embeddings(embeddings);

    // Generate tokens IDs
    int *tokenIDs = (int*)calloc(sizeof(int), VOCAB_SIZE);
    fill_token_IDs(tokenIDs);

    // Get the embeddings for the token sequence
    float *input_embeddings = (float*)calloc(sizeof(float), EMBEDDING_SIZE * SEQUENCE_LENGTH);
    get_embeddings(tokenIDs, SEQUENCE_LENGTH, embeddings, input_embeddings);

    // Initialize output matrix for concatenated attention matrices
    float *concat_attnts = (float*)calloc(sizeof(float), EMBEDDING_SIZE * SEQUENCE_LENGTH * NUM_HEADS);

    // Calculate attention score matrix for each head
    for (int i = 0; i < NUM_HEADS; i++){

        // Fill weight matrices 
        float *Q_weights = (float*)calloc(sizeof(float), EMBEDDING_SIZE * QKV_MATRIX_WIDTH);
        float *K_weights = (float*)calloc(sizeof(float), EMBEDDING_SIZE * QKV_MATRIX_WIDTH);
        float *V_weights = (float*)calloc(sizeof(float), EMBEDDING_SIZE * QKV_MATRIX_WIDTH);
        fill_QKV_weights(Q_weights, K_weights, V_weights);

        // Calculate QKV matrices using input stationary dataflow
        float *Q = (float*)calloc(sizeof(float), SEQUENCE_LENGTH * QKV_MATRIX_WIDTH);
        float *K = (float*)calloc(sizeof(float), SEQUENCE_LENGTH * QKV_MATRIX_WIDTH);
        float *V = (float*)calloc(sizeof(float), SEQUENCE_LENGTH * QKV_MATRIX_WIDTH);
        fill_QKV_matrices(input_embeddings, Q_weights, K_weights, V_weights, Q, K, V);

        // Calculate attention matrix for the current head
        float *output = (float*)calloc(sizeof(float), SEQUENCE_LENGTH * EMBEDDING_SIZE);
        attention(Q, K, V, output, SEQUENCE_LENGTH, EMBEDDING_SIZE);

        // Concatenate attention matrix output
        for (int j = 0; j < SEQUENCE_LENGTH; j++){
            for (int k = 0; k < EMBEDDING_SIZE; k++){
                concat_attnts[k + (j * EMBEDDING_SIZE * NUM_HEADS) + (i * EMBEDDING_SIZE)] = output[j * EMBEDDING_SIZE + k];
            }
        }
        // Free data
        free(Q_weights);
        free(K_weights);
        free(V_weights);
        free(Q);
        free(K);
        free(V);
    }

    // Create output weight matrix
    float *output_weights = (float*)calloc(sizeof(float), EMBEDDING_SIZE * NUM_HEADS * FINAL_DIMENSIONALITY);
    fill_output_weights(output_weights);

    // Perform final linear transform
    float *result = (float*)calloc(sizeof(float), SEQUENCE_LENGTH * FINAL_DIMENSIONALITY);
    final_transform(result, concat_attnts, output_weights);

    // Free data
    free(embeddings);
    free(tokenIDs);
    free(input_embeddings);
    free(concat_attnts);
    free(output_weights);
    free(result);

    // DONE
}


int main(int argc, char* argv[]) {

  // Set up output file
  // file format: num_embedings, speedup, time_opt, time_unopt
  std::ofstream resultFile("result.txt");
  resultFile << "file format: num_embedings, speedup, time_opt, time_unopt" << std::endl;

  // Time the transformer
  int trials = atoi(argv[1]);
  std::chrono::duration<double, std::milli> difference_sum(0);
  std::chrono::duration<double, std::milli> unopt_sum(0);
  std::chrono::duration<double, std::milli> opt_sum(0);
  for (int i = 0; i < trials; i++){
    std::chrono::duration<double, std::milli> unopt_time = time_transformer(0); // Time unoptimized transformer
    std::chrono::duration<double, std::milli> opt_time = time_transformer(1); // Time optimized transformer
    std::chrono::duration<double, std::milli> difference = (opt_time - unopt_time);
    difference_sum += difference;
    unopt_sum += unopt_time;
    opt_sum += opt_time;
    resultFile << std::fixed << EMBEDDING_SIZE << ", " << std::setprecision(4) << difference.count() 
    << " ms" << ", " << opt_time.count() << " ms" << ", " << unopt_time.count() << " ms" << std::endl;
    std::cout << "Trial " << i + 1 << " of " << trials << " done." << std::endl;
  }
  std::chrono::duration<double, std::milli> avg_diff = difference_sum / trials;
  double factor = unopt_sum / opt_sum;
  resultFile << "avg speedup time: " << std::fixed << std::setprecision(4) << avg_diff.count() << " ms" << std::endl;
  resultFile << "avg speedup factor: " << std::fixed << factor << "x" << std::endl;

return 0;

}
