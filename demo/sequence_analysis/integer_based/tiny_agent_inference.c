#include "../../../src/integer_based/tiny_agent.h"

#define GRU_HIDDEN_SIZE 4
#define GRU_INPUT_SIZE 1
#define FC1_INPUT_SIZE 4
#define FC2_INPUT_SIZE 4
#define PREDICTION_SIZE 2

#define GRU_LOAD_PATH "./data/gru.ta"
#define MLP_LOAD_PATH "./data/mlp.ta"
#define MAX_SEQ_LENGTH 20

#define FLOAT_RANGE 10 // all values belongs to [-10, 10]
#define SCALE_BIT  30
#define SCALE (FLOAT_RANGE / (2 ** SCALE_BIT))
#define TABLE_RANGE 10000

#define SIGMOID_INPUT "./data/sigmoid_table_input.ta"
#define SIGMOID_OUTPUT "./data/sigmoid_table_output.ta"
#define TANH_INPUT "./data/tanh_table_input.ta"
#define TANH_OUTPUT "./data/tanh_table_output.ta"

// convert a quantized value back to its floating-point version
double r_quantized(TYPE v)
{
    return (double) v * FLOAT_RANGE / (1 << SCALE_BIT);
}

// quantized a floating-point based value into its integer-based version
TYPE f_quantized(float v)
{
	return v * (1 << SCALE_BIT) / FLOAT_RANGE;
}


int main()
{
    int i, j;

    // # don't generate sequence length > MAX_SEQ_LENGTH
    float input_0[] = {1, 0, 1, 1, 0, 0, 1, 0, 1};
    float input_1[] = {0, 0, 1, 0, 1};
    float input_2[] = {1, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    float *data_to_test[] = {input_0, input_1, input_2};
    int data_to_test_len = 3;
    int seq_len[] = {9, 5, 10};

    init_sigmoid_table(SIGMOID_INPUT, SIGMOID_OUTPUT);
    init_tanh_table(TANH_INPUT, TANH_OUTPUT);

    struct gru_cell cell;
    init_gru_cell(&cell, GRU_HIDDEN_SIZE, GRU_INPUT_SIZE);
    cell.read(&cell, GRU_LOAD_PATH);

    struct fc_layer layer1, layer2;
    init_fc_layer(&layer1, FC1_INPUT_SIZE, FC2_INPUT_SIZE, SIGMOID);
    init_fc_layer(&layer2, FC2_INPUT_SIZE, PREDICTION_SIZE, NONE);

    struct mlp nn;
    init_mlp(&nn);
    nn.add_layer(&nn, &layer1);
    nn.add_layer(&nn, &layer2);
    nn.read(&nn, MLP_LOAD_PATH);

    printf("[+] read param from:\n");
    printf("[+] \t %s\n", GRU_LOAD_PATH);
    printf("[+] \t %s\n", MLP_LOAD_PATH);
    printf("[+]\n");

    for (i = 0; i < data_to_test_len; i++) {
        float *input = data_to_test[i];

        cell.clear(&cell);
        for (j = 0; j < seq_len[i]; j++) {
            float float_based = input[j];
            TYPE integer_based = f_quantized(float_based);
            cell.inference(&cell, &integer_based);
        }
        nn.inference(&nn, cell.hidden);

        printf("[+] now testing sequence: [");
        for (j = 0; j < seq_len[i]; j++) {
            printf("%d", (int) input[j]);
            if (j != seq_len[i] - 1) {
                printf(", ");
            }
        }
        printf("]\n");

        float float_based_nn_predict_0 = r_quantized(nn.result[0]);
        float float_based_nn_predict_1 = r_quantized(nn.result[1]);

        printf("[+] \t prediction: [%.4f, %.4f] -> number of 1: %.0f | length of sequence: %.0f \n", 
            float_based_nn_predict_0, float_based_nn_predict_1, 
            (float_based_nn_predict_0 * seq_len[i]), (float_based_nn_predict_1 * MAX_SEQ_LENGTH));
    }

    free_sigmoid_table();
    free_tanh_table();

    cell.free(&cell);
    nn.free(&nn);
}
